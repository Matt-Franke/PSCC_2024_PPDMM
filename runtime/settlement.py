"""
This MPC protocol imports and verifies the MC balances.
It then computes the imbalances and the resulting balance updates.
"""
from mpyc.runtime import mpc
from mpyc import asyncoro
from mpyc import sectypes
from mpyc import thresha
from mpyc import finfields
import json
import numpy
import pickle
from mpyc.gmpy import is_prime, isqrt
import random
import os

nodes = len(mpc.parties)
lb = 64
decimals = 8
secfxp = mpc.SecFxp(lb)
lb_int = nodes.bit_length()
secint = mpc.SecInt()

# ADD in, fix later
returnType = asyncoro.returnType
Future = asyncoro.Future
SecureObject = sectypes.SecureObject
DEBUG = True


def sign(a):
    if a < 0:
        return -1
    if a > 0:
        return 1
    return 0


@mpc.coroutine
async def f_sqrt(val):
    """
    My sqrt function based on the protected function _fsqrt in the statistics module of MPyC
    https://github.com/lschoe/mpyc/blob/39d0a1052b0bf153b95e58bee97dc874ef4c4aed/mpyc/statistics.py#L200
    :param val the single input
    :return: the sqrt of that term
    """
    sectype = type(val)
    f = sectype.frac_length
    e = (sectype.bit_length + f - 1) // 2  # (l+f)/2 - f = (l-f)/2 in [0..l/2]
    r = sectype(0)
    j = 2 ** (e - f)
    for _ in range(e + 1):
        h = r + j
        r = mpc.if_else(h * h <= val, h, r)
        j /= 2
    return r


@mpc.coroutine
async def f_norm(seclist):
    """
    My own norm function based on the protected function _fsqrt in the statistics module of MPyC
    https://github.com/lschoe/mpyc/blob/39d0a1052b0bf153b95e58bee97dc874ef4c4aed/mpyc/statistics.py#L200
    :param seclist: the list of squared inputs to have the norm returned from
    :return: f_norm, the fixed point norm
    """
    sq_sum = mpc.sum(seclist)
    r = f_sqrt(sq_sum)

    return r, sq_sum


@mpc.coroutine
async def add_2(x, y):
    """Secure addition of vectors x and y."""
    if not x:
        return []

    x, y = x[:], y[:]
    stype = type(x[0])  # all elts assumed of same type
    n = len(x)
    if not stype.frac_length:
        await returnType(stype, n)
    else:
        y0_integral = isinstance(y[0], int) or isinstance(y[0], SecureObject) and y[0].integral
        await returnType((stype, x[0].integral and y0_integral), n)

    x, y = await mpc.gather(x, y)
    for i in range(n):
        x[i] = x[i] + y[i]
    return x


@mpc.coroutine
async def add_3(x, y, z):
    """Secure addition of vectors x and y."""
    if not x:
        return []

    x, y, z = x[:], y[:], z[:]
    stype = type(x[0])  # all elts assumed of same type
    n = len(x)
    if not stype.frac_length:
        await returnType(stype, n)
    else:
        y0_integral = isinstance(y[0], int) or isinstance(y[0], SecureObject) and y[0].integral
        await returnType((stype, x[0].integral and y0_integral), n)

    x, y, z = await mpc.gather(x, y, z)
    for i in range(n):
        x[i] = x[i] + y[i] + z[i]
    return x


def gen_rounding(in_data, decs: int):
    """
    A generic rounding function that makes sure all elements in the input are rounded to the same number of decimals
    We are expecting these to floats
    """
    if not in_data:
        return 0.0

    if not isinstance(in_data, list):
        # just a single float
        output = round(in_data, decs)
    else:
        if not isinstance(in_data[0], list):
            # just a simple list of floats
            output = [round(j, decs) for j in in_data]
        else:
            if not isinstance(in_data[0][0], list):
                # a list of lists of floats
                output = [[round(j, decs) for j in ll] for ll in in_data]
            else:
                # a list of lists of lists of floats
                output = [[[round(j, decs) for j in kk] for kk in ll] for ll in in_data]

    return output


def get_ab(n: int):
    """
    For a number of vectors n, get a (number of sum3s) and b (number of sum2s)
    :param n:
    :return: a, b
    """
    z = n % 3
    low = n // 3

    if z == 0:
        a = low
        b = 0
    elif z == 1:
        a = low - 1
        b = 2
    else:
        a = low
        b = 1

    return a, b


@mpc.coroutine
async def n_vector_add(vectors, np=False):
    """
    Secure addition of n Shamir secret shared vectors
    All vectors should have the same size!!!
    :param vectors: list of lists of secure numbers or list of secure arrays
    :return:
    """
    if not vectors:
        return []

    if np:
        # if even, just add them all up, if odd add a zero vector of same length to make it work
        raise NotImplemented

    else:
        n = len(vectors)  # at least 1
        vec_n = len(vectors[0])  # at least 1
        s_type = type(vectors[0][0])

        check = sum(1 if len(vectors[i]) == vec_n else 0 for i in range(n))
        if check != n:
            # one of the vectors is not the same length!
            raise ValueError
        if DEBUG:
            print("Begin with {} vectors".format(n))

        while n > 3:
            a, b = get_ab(n)
            results = [s_type(0.0)] * (a + b)
            if DEBUG:
                print("Combine {} vectors with {} sum3s and {} sum2s".format(n, a, b))
            for i in range(a):
                if DEBUG:
                    print("Sum {} with vectors {}, {}, {}".format(i, 3 * i, 3 * i + 1, 3 * i + 2))
                results[i] = add_3(vectors[3 * i], vectors[3 * i + 1], vectors[3 * i + 2])

            for j in range(b):
                if DEBUG:
                    print("Sum {} with vectors {}, {}".format(a + j, 3 * a + 2 * j, 3 * a + 2 * j + 1))
                results[a + j] = add_2(vectors[3 * a + 2 * j], vectors[3 * a + 2 * j + 1])

            vectors = results[:]
            n = a + b
            if DEBUG:
                print("Continue with {} vectors".format(n))

    if n == 1:
        return vectors[0]
    if n == 2:
        return add_2(vectors[0], vectors[1])
    if n == 3:
        return add_3(vectors[0], vectors[1], vectors[2])


@mpc.coroutine
async def recover_from_shares(x, shares):
    """
    Based on the output function from MPyC just starting with shares already gathered.
    Output to all with the default threshold, since this is used after commitments are opened
    x is the secure object to be recovered
    share is this nodes clear text share of said object

    X will always be just one number, we will recover arrays as individual elements due to commitments

    """
    x_is_list = isinstance(x, list)
    if x_is_list:
        x = x[:]
    else:
        x = [x]
    await mpc.returnType(Future)

    n = len(x)
    if not n:
        return []

    t = mpc.threshold
    m = len(mpc.parties)
    receivers = list(range(m))  # default
    sftype = type(x[0])  # all elts assumed of same type

    # here is the only switch. We will not gather since we already have done that
    field = x[0].field

    x = shares[:]  # no need to gather, we have done that previously
    recombine = thresha.recombine
    marshal = field.to_bytes
    unmarshal = field.from_bytes

    # Send share x to all successors in receivers.
    share = None
    for peer_pid in receivers:
        if 0 < (peer_pid - mpc.pid) % m <= t:
            if share is None:
                share = marshal(x)
            mpc._send_message(peer_pid, share)

    # Receive and recombine shares if this party is a receiver.
    if mpc.pid in receivers:
        shares = [mpc._receive_message((mpc.pid - t + j) % m) for j in range(t)]
        shares = await mpc.gather(shares)
        points = [((mpc.pid - t + j) % m + 1, unmarshal(shares[j])) for j in range(t)]
        points.append((mpc.pid + 1, x))
        y = recombine(field, points)
        y = [field(a) for a in y]

        if issubclass(sftype, mpc.SecureObject):
            f = sftype._output_conversion
            if f is not None:
                y = [f(a) for a in y]
    else:
        y = [None] * n

    if not x_is_list:
        y = y[0]
    return y


@mpc.coroutine
async def reshare_from_shares(fld, typ, vals):
    """
    :param fld: The field the shares were in originally (called via GF(modulus))
    :param typ: The secure type (secint) the value(s) was/were shared as originally
    :param vals: the integer values of our shares
    """

    x_is_list = isinstance(vals, list)
    if not x_is_list:
        x = [vals]
    else:
        x = vals[:]

    if x == []:
        return []

    sftype = typ  # all elts assumed of same type
    if issubclass(sftype, mpc.SecureObject):
        rettype = sftype
        if x_is_list:
            await mpc.returnType(rettype, len(x))
        else:
            await mpc.returnType(rettype)
    else:
        await mpc.returnType(Future)

    t = mpc.threshold

    field = fld

    m = len(mpc.parties)

    shares = thresha.random_split(field, x, t, m)
    shares = [field.to_bytes(elts) for elts in shares]

    # Recombine the first 2t+1 output_shares.
    shares = mpc._exchange_shares(shares)
    shares = await mpc.gather(shares[:2 * t + 1])

    points = [(j + 1, field.from_bytes(s)) for j, s in enumerate(shares)]
    y = thresha.recombine(field, points)

    y = [field(a) for a in y]

    if not x_is_list:
        y = y[0]
    return y


async def keygen(g):
    """Threshold ElGamal key generation."""
    group = type(g)
    secgrp = mpc.SecGrp(group)
    n = group.order
    if n is not None and is_prime(n):
        secnum = mpc.SecFld(n)
    else:
        l = isqrt(-group.discriminant).bit_length()
        secnum = mpc.SecInt(l)

    while True:
        x = mpc._random(secnum)
        h = await secgrp.repeat_public(g, x)  # g^x
        if h != group.identity:
            # NB: this branch will always be followed unless n is artificially small
            return x, h


def encrypt(g, h, M, r):
    """ElGamal encryption."""
    # randomness now generated before hand
    c = (g ^ r, (h ^ r) @ M)
    return c


def convert_commits(C: list):
    """Convert a list of EC commits to a list of tuples of JSON-able values"""
    out = []
    for c in C:
        a = c[0].value
        b = c[1].value

        a_s = [aa.value for aa in a]
        b_s = [bb.value for bb in b]
        out.append((a_s, b_s))

    return out


def compare_commits(C1: tuple, C2: tuple):
    """Convert a list of tuples of JSON-able values to a a list of EC commits"""
    C1a = C1[0]
    C1b = C1[1]
    C2a = C2[0]
    C2b = C2[1]

    check_a = sum(1 if a1 == a2 else 0 for a1, a2 in zip(C1a, C2a))
    check_b = sum(1 if b1 == b2 else 0 for b1, b2 in zip(C1b, C2b))

    if check_a+check_b == 6:
        return 1
    else:
        return 0


# Load in the up and downsets, i.e. the topology data we need
with open(os.path.normpath("./runtime/temp/imb_prices.json"), 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

up = json_object["up"]
dw = json_object["dw"]
spot = json_object["spot"]
tariff = json_object["tariff"]
MODE = json_object["MODE"]
res_globals = os.path.normpath("./runtime/temp/imb.json")


async def main():
    """
    This is our MPC code
    :return:
    """
    await mpc.start()  # connect to all other parties

    print("Executing with {} nodes with threshold {}".format(nodes, mpc.threshold))
    """
    Input
    """
    # Get prices
    U = list(map(secfxp, up))
    D = list(map(secfxp, dw))
    S = list(map(secfxp, spot))

    # Get and input actual net injection, forecasted error and scheduled ni and alphas
    path_state = os.path.normpath("./runtime/states/" + str(mpc.pid) + ".json")

    with open(path_state, 'r') as openfile:
        # Reading from json file
        input_dict = json.load(openfile)

    ni_act = input_dict["ni_act"]
    ni_sched = input_dict["ni_sched"]
    Delta = input_dict["Delta"]
    num_time = input_dict["num_time"]

    N_A = mpc.input(list(map(secfxp, ni_act)))
    N_S = mpc.input(list(map(secfxp, ni_sched)))
    T = secfxp(tariff)
    signD = [sign(d) for d in Delta]  # this is public anyways

    """
    Recover and confirm balances
    """
    path_dvs = os.path.normpath("./runtime/temp/DVS_" + str(mpc.pid) + ".json")
    with open(path_dvs, 'r') as openfile:
        # Reading from json file
        dvs_dict = json.load(openfile)

    B_share = dvs_dict["B_share"]
    B_flag = dvs_dict["B_flag"]
    B_mod = dvs_dict["B_mod"]
    C_list = dvs_dict["C_list"]
    r = dvs_dict["r"]
    B_act = dvs_dict["B_act"]
    x_mod = dvs_dict["x_mod"]
    x_val = dvs_dict["x_val"]

    # PKI
    secgrp = mpc.SecEllipticCurve('secp256k1', 'projective') # Version 0.8.15 or higher
    # Every party generates a key pair and publishes their public key
    group = secgrp.group
    g = group.generator  # same for everyone and same as before!

    # Reestablish public keys from last time
    x_field = mpc.SecFld(x_mod)
    x_recov = x_field(x_val)
    h_prime = await secgrp.repeat_public(g, x_recov)  # g^x
    h = await mpc.transfer(h_prime)

    # Input our own balance
    B_type = secint if B_flag else secfxp
    B_open_sec = mpc.input(B_type(B_act))

    # Everyone loads their shares of balances and reshares them
    B_fld = finfields.GF(B_mod)
    B_rec_sec = reshare_from_shares(B_fld, B_type, B_share)

    # DVS Check 1: Are these two values the same?
    target = 0 if B_flag else 0.0
    diff = mpc.vector_sub(B_open_sec, B_rec_sec)
    diff_check = [mpc.if_else(d == target, 1, 0) for d in diff]
    check_sec = await mpc.output(diff_check)

    failed_first = [j for j in range(nodes) if not check_sec[j]]  # if you failed the secure check
    # If you did not fail, you do not open. So we have you open (0,0)
    if mpc.pid not in failed_first:
        exp = 0
        r = 0
    exps = await mpc.transfer(exp)
    B_open = [e * pow(10, -1 * decimals) for e in exps]
    r_open = await mpc.transfer(r)
    # All parties verify the shared value by recalculating the commit blobs via encrypt calls
    C_prime = [encrypt(g, h[k], g ^ exps[k], r_open[k]) for k in range(nodes)]
    C_prime_list = convert_commits(C_prime)
    # Consensus to ensure we agree all commits where correct opened
    check_C = [1 if k not in failed_first or compare_commits(C_list[k], C_prime_list[k]) else 0 for k in range(nodes)]

    # Check if reconstructed and commited value are also the same
    B_rec_sec_out = [await mpc.output(B_rec_sec[j]) if j in failed_first else 0 for j in range(nodes)]
    B_recon = [round(b, decimals) for b in B_rec_sec_out]
    check_B = [1 if k not in failed_first or B_recon[k] == B_open[k] else 0 for k in range(nodes)]

    # DVS Level 2:
    if sum(check_B) > mpc.threshold:  # if what was reconstructed, is what was opened
        if sum(check_C) > mpc.threshold:  # if what was opened is what was committed
            ans = 1
    else:
        ans = 0

    sum_ans = mpc.sum(mpc.input(secint(ans)))
    DVS_dec = await mpc.output(mpc.if_else(sum_ans > mpc.threshold, 1, 0))

    bals_mpc = [B_rec_sec[i] if i in failed_first else B_open_sec[i] for i in range(nodes)]

    """
    Calculate imbalances
    """
    E_A = [[secfxp(0.0)] * num_time] * nodes

    for i in range(nodes):
        e = mpc.vector_sub(N_A[i], N_S[i])
        E_A[i] = e

    """
    Calculate imbalance payments
    """
    i_bals = [secfxp(0.0)] * nodes
    for t in range(num_time):
        for i in range(nodes):
            if i:  # skip feeder
                totD_sign = signD[t]  # sign of total imbalance
                if totD_sign == 1:
                    # network in overall surplus
                    i_bals[i] += mpc.if_else(E_A[i][t] > 0, E_A[i][t] * (D[t] - T), E_A[i][t] * (
                            S[t] + T))
                elif totD_sign == -1:
                    # network in overall deficit
                    i_bals[i] += mpc.if_else(E_A[i][t] > 0, E_A[i][t] * (S[t] - T), E_A[i][t] * (
                            U[t] + T))
                else:
                    # network balanced
                    pass

    """
    Update Balances
    """
    f_bals = mpc.vector_add(bals_mpc, i_bals)
    final_balance = await mpc.output(f_bals, 0)

    """
    Output Balances
    """
    if mpc.pid == 0:
        result_dict = {"balances": final_balance, "DVS":DVS_dec}

        result_string = json.dumps(result_dict)
        with open(res_globals, "w") as outfile:
            outfile.write(result_string)

    await mpc.shutdown()  # disconnect, but only once all other parties reached this point


if __name__ == '__main__':
    mpc.run(main())
