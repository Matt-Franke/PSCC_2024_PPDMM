"""
This MPC protocol is for the normal operation each iteration
TOTAL OF, new globals, termination decision
"""
from mpyc.runtime import mpc
from mpyc import asyncoro
from mpyc import sectypes
from mpyc import thresha
import json
import numpy
from mpyc.gmpy import is_prime, isqrt
import random
import os

nodes = len(mpc.parties)
lb = 64
decimals = 8
secfxp = mpc.SecFxp(lb)
lb_int = nodes.bit_length()
secint = mpc.SecInt(lb_int)

# ADD in, fix later
returnType = asyncoro.returnType
SecureObject = sectypes.SecureObject
Future = asyncoro.Future


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
async def n_vector_add(vectors):
    """
    Secure addition of n Shamir secret shared vectors
    All vectors should have the same size!!!
    :param vectors: list of lists of secure numbers or list of secure arrays
    :param nps: whether we are dealing with list of sec n or arrays
    :return:
    """
    if not vectors:
        return []
    n = len(vectors)  # at least 1
    vec_n = len(vectors[0])  # at least 1
    s_type = type(vectors[0][0])

    check = sum(1 if len(vectors[i]) == vec_n else 0 for i in range(n))
    if check != n:
        # one of the vectors is not the same length!
        raise ValueError
    while n > 3:
        a, b = get_ab(n)
        results = [s_type(0.0)] * (a + b)
        for i in range(a):
            results[i] = add_3(vectors[3 * i], vectors[3 * i + 1], vectors[3 * i + 2])

        for j in range(b):
            results[a + j] = add_2(vectors[3 * a + 2 * j], vectors[3 * a + 2 * j + 1])

        vectors = results[:]
        n = a + b

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
    if DEBUG:
        print("Recovering x which is a list: {}".format(x_is_list))
    if x_is_list:
        x = x[:]
    else:
        x = [x]
    await mpc.returnType(Future)

    n = len(x)
    if not n:
        return []
    if DEBUG:
        print("Recovering {} values".format(n))

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
    if DEBUG:
        print("Sending out my shares")
    share = None
    for peer_pid in receivers:
        if 0 < (peer_pid - mpc.pid) % m <= t:
            if share is None:
                share = marshal(x)
            mpc._send_message(peer_pid, share)
            if DEBUG:
                print("Node {} sending share to Node {}".format(mpc.pid, peer_pid))
    if DEBUG:
        print("All my shares send out")
    # Receive and recombine shares if this party is a receiver.
    if mpc.pid in receivers:
        shares = [0] * t
        if DEBUG:
            print("Shares {}, len: {}".format(shares, len(shares)))
        for j in range(t):
            from_who = (mpc.pid - t + j) % m
            shares[j] = mpc._receive_message(from_who)
            if DEBUG:
                print("Node {} sending share to Node {}".format(from_who, mpc.pid))

        if DEBUG:
            print("All my shares received")
        shares = await mpc.gather(shares)
        if DEBUG:
            print("All my shares gathered")
        points = [((mpc.pid - t + j) % m + 1, unmarshal(shares[j])) for j in range(t)]
        points.append((mpc.pid + 1, x))
        y = recombine(field, points)
        y = [field(a) for a in y]
        if DEBUG:
            print("All the outputs recovered")
        if issubclass(sftype, mpc.SecureObject):
            f = sftype._output_conversion
            if f is not None:
                y = [f(a) for a in y]
    else:
        y = [None] * n

    if DEBUG:
        print("Outputting {}".format(y))
    if not x_is_list:
        y = y[0]
    return y


def send_receive():
    t = mpc.threshold
    m = len(mpc.parties)
    receivers = list(range(m))  # default
    to_list = []
    from_list = []

    for peer_pid in receivers:
        if 0 < (peer_pid - mpc.pid) % m <= t:
            to_list.append(peer_pid)

    # Receive and recombine shares if this party is a receiver.
    if mpc.pid in receivers:
        from_list = [(mpc.pid - t + j) % m for j in range(t)]

    return to_list, from_list


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


# Load in the up and downsets, i.e. the topology data we need
with open(os.path.normpath("./runtime/temp/parameters.json"), 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

up_set = json_object["up_set"]
dw_set = json_object["dw_set"]
MODE = json_object["MODE"]
DEBUG = json_object["DEBUG"]

OP = json_object["OP"]  # 0 is normal, 1 is decision, 2 is balance
da_base = json_object["da"]
flex_base = json_object["flex"]
res_globals = os.path.normpath("./runtime/temp/res.json")


async def main():
    """
    This is our MPC code
    :return:
    """
    await mpc.start()  # connect to all other parties
    print("Working with {} parties in MODE {} and OP {}".format(nodes, MODE, OP))
    """
    Input
    """
    path_state = os.path.normpath("./runtime/states/" + str(mpc.pid) + ".json")

    with open(path_state, 'r') as openfile:
        # Reading from json file
        input_dict = json.load(openfile)

    c_i = input_dict["c_i"]
    fP_scaled = input_dict["fP_scaled"]
    fQ_scaled = input_dict["fQ_scaled"]
    us_scaled = input_dict["us_scaled"]
    rho_resP = input_dict["rho_resP"]
    num_time = input_dict["num_time"]
    ni = input_dict["ni"]
    rho = input_dict["rho"]

    if MODE:
        # alpha will have very different shape in Mode 4
        alpha_unscaled = input_dict["alpha"]  # not technically correct but it might work
        alpha_scaled = input_dict["alpha_scaled"]
        rho_resA = input_dict["rho_resA"]
        shareSD = input_dict["shareSD"]

    if MODE == 3:
        rho_v_scaled = input_dict["rho_v_scaled"]
        rho_f_scaled = input_dict["rho_f_scaled"]

    path_it = os.path.normpath("./runtime/temp/" + str(mpc.pid) + ".json")

    with open(path_it, 'r') as openfile:
        # Reading from json file
        input_dict = json.load(openfile)

    if OP in [0, 1]:
        r_i = input_dict["r_i"]
        s_i = input_dict["s_i"]
        de_f = input_dict["de_f"]
        neg_de_f = input_dict["neg_de_f"]
        de_s = input_dict["de_s"]  # Switch to de_s once actually implementing

    if DEBUG:
        print("Reading inputs done")

    """
    Initialisation. We work with lists of secure arrays
    """

    N = mpc.input(list(map(secfxp, ni)))
    local_resP = mpc.input(list(map(secfxp, rho_resP)))

    if OP in [0, 2]:
        D = [list(map(secfxp, da_i)) for da_i in da_base]

    if MODE in [1, 2, 3] and OP in [0, 2]:
        F = list(map(secfxp, flex_base))

    if MODE == 4 and OP in [0, 2]:
        F = [list(map(secfxp, fl_i)) for fl_i in flex_base]
        raise NotImplemented

    if MODE and OP == 2:
        S = [mpc.input(secfxp(shareSD[t])) for t in range(num_time)]
        A_T = [mpc.input(secfxp(alpha_unscaled[t])) for t in range(num_time)]

    zero = [secfxp(0.0)] * num_time
    ones = [secfxp(1.0)] * num_time
    ones_n = [[secfxp(1.0)] * num_time] * nodes

    if DEBUG:
        to_l, from_l = send_receive()
        print("Node {}. Send to: {}. Receive from {}.".format(mpc.pid, to_l, from_l))

    """
    SMPC 1: Calculate Combined OF
    Normal
    """
    if OP == 0:
        if DEBUG:
            print("SMPC 1 Output")

        zC = await mpc.output(mpc.sum(mpc.input(secfxp(c_i))), 0)
    else:
        zC = 0

    if DEBUG:
        print("SMPC 1 Done")

    """
    SMPC 2: Calculate global variables
    Normal and Balance
    """
    if OP in [0, 2]:
        # Everyone's individual values as a secure array of (nodes *) nodes * time
        fP_s = [list(map(secfxp, fP_i)) for fP_i in fP_scaled]
        fQ_s = [list(map(secfxp, fQ_i)) for fQ_i in fQ_scaled]
        us_s = [list(map(secfxp, us_i)) for us_i in us_scaled]

        if MODE in [1, 2, 3]:
            alpha_s = [list(map(secfxp, al_i)) for al_i in alpha_scaled]

        if MODE == 3:
            rho_v_s = [list(map(secfxp, rv_i)) for rv_i in rho_v_scaled]
            rho_f_s = [list(map(secfxp, rf_i)) for rf_i in rho_f_scaled]

        if MODE == 4:
            alpha_s = [[list(map(secfxp, al_k)) for al_k in al_i] for al_i in alpha_scaled]
            raise NotImplemented

        # Gather Inputs for Variables per Reference Node. Lists of lists of arrays
        fP = [mpc.input(fP_s[ref_n]) for ref_n in range(nodes)]
        fQ = [mpc.input(fQ_s[ref_n]) for ref_n in range(nodes)]
        us = [mpc.input(us_s[ref_n]) for ref_n in range(nodes)]

        if MODE in [1, 2, 3]:
            alpha = [mpc.input(alpha_s[ref_n]) for ref_n in range(nodes)]

        if MODE == 3:
            rho_v = [mpc.input(rho_v_s[ref_n]) for ref_n in range(nodes)]
            rho_f = [mpc.input(rho_f_s[ref_n]) for ref_n in range(nodes)]

        if MODE == 4:
            # alpha = [[mpc.input(alpha_s[ui][ref_n]) for ref_n in range(nodes)] for ui in range(nodes)]
            raise NotImplemented

        # Calculation of Global Variables
        P = [n_vector_add(fP[ref_n]) for ref_n in range(nodes)]
        Q = [n_vector_add(fQ[ref_n]) for ref_n in range(nodes)]
        U = [n_vector_add(us[ref_n]) for ref_n in range(nodes)]

        if MODE in [1, 2, 3]:
            A = [n_vector_add(alpha[ref_n]) for ref_n in range(nodes)]

        if MODE == 3:
            RV = [n_vector_add(rho_v[ref_n]) for ref_n in range(nodes)]
            RF = [n_vector_add(rho_f[ref_n]) for ref_n in range(nodes)]

        if MODE == 4:
            # A = [[n_vector_add(alpha[ui][ref_n]) for ref_n in range(nodes)] for ui in range(nodes)]
            raise NotImplemented

        if DEBUG:
            print("SMPC 2 Output")
        # Output as lists of lists
        zP = [await mpc.output(P[ref_n], 0) for ref_n in range(nodes)]
        zQ = [await mpc.output(Q[ref_n], 0) for ref_n in range(nodes)]
        zU = [await mpc.output(U[ref_n], 0) for ref_n in range(nodes)]

        if not MODE:
            zA = 0
            zRV = 0
            zRF = 0

        if MODE in [1, 2, 3]:
            zA = [await mpc.output(A[ref_n], 0) for ref_n in range(nodes)]
            zRV = 0
            zRF = 0

        if MODE == 3:
            zRV = [await mpc.output(RV[ref_n], 0) for ref_n in range(nodes)]
            zRF = [await mpc.output(RF[ref_n], 0) for ref_n in range(nodes)]

        if MODE == 4:
            # zA = [[await mpc.output(A[ui][ref_n], 0) for ref_n in range(nodes)] for ui in range(nodes)]
            # it definitely does not like this lol
            raise NotImplemented
    else:
        zP = 0
        zQ = 0
        zU = 0
        zA = 0
        zRV = 0
        zRF = 0

    if DEBUG:
        print("SMPC 2 Done")

    """
    SMPC 3: Calculate global prices
    Normal and Balance
    """
    if OP in [0, 2]:
        # Calculate
        for n in range(nodes):
            # Day Ahead Updated
            if n != 0:
                # our global line flow
                global_resP = P[n]
            else:
                # whatever flowed into Node 0 flows onwards, so just take the fP of the next node
                global_resP = mpc.vector_sub(zero, n_vector_add(
                [P[k] for k in dw_set[n] if k != n] + [zero]))

            rho_gP = mpc.scalar_mul(secfxp(rho), global_resP)
            # if rho gets too big, things do break (so increase bit size)
            diff = mpc.vector_sub(local_resP[n], rho_gP)
            D[n] = mpc.vector_add(D[n], diff)

        zD = [await mpc.output(D[ref_n], 0) for ref_n in range(nodes)]
        if DEBUG:
            print("SMPC 3 Output of D")

        # Flexibility Updated
        if MODE in [1, 2, 3]:
            local_A = n_vector_add(mpc.input(list(map(secfxp, alpha_unscaled))))  # just add everyone's own part. factors
            global_A = n_vector_add(A)  # add everyone's global part factors
            F_diff = mpc.vector_sub(local_A, global_A)  # so far this is the sum of the lambda_A's
            F = mpc.vector_sub(F, F_diff)  # since we want the negative of this as the shadow price
        elif MODE == 4:
            raise NotImplemented
        else:
            F = 0

        if MODE in [1, 2, 3]:
            # Get Outputs
            if DEBUG:
                print("SMPC 3 Output of F")
            zF = await mpc.output(F, 0)
        elif MODE == 4:
            raise NotImplemented
        else:
            zF = 0
    else:
        zD = 0
        zF = 0

    if DEBUG:
        print("SMPC 3 Done")

    """
    SMPC 4: Evaluate Stopping Criteria
    Normal and Decision
    """
    if OP in [0, 1]:
        r_sum = mpc.sum(mpc.input(secint(r_i)))
        s_sum = mpc.sum(mpc.input(secint(s_i)))

        # N from above, a n*t array
        pDEF = mpc.sum(mpc.input(secfxp(de_f)))  # Per Timestep Surplus
        nDEF = mpc.sum(mpc.input(secfxp(neg_de_f)))  # Per Timestep Surplus
        DES = mpc.sum(mpc.input(secfxp(de_s)))  # Total Surplus

        res = n_vector_add(N)  # Per Timetstep
        tot_res = mpc.sum(res)  # Total

        if OP == 1:
            # Per Timestep - Only when confirming
            p_f = mpc.all([mpc.if_else(res[n] < pDEF, 1, 0) for n in range(nodes)])
            n_f = mpc.all([mpc.if_else(nDEF < res[n], 1, 0) for n in range(nodes)])
            f_tot = await mpc.output(mpc.all([p_f, n_f]), 0)
        else:
            f_tot = 0

        # Total
        small_delta = secfxp(0.00000001)
        p_Total = tot_res - DES - small_delta
        n_Total = small_delta - DES - tot_res

        if DEBUG:
            print("SMPC 4 Output")
        r_tot = await mpc.output(r_sum, 0)
        s_tot = await mpc.output(s_sum, 0)
        p_tot = await mpc.output(p_Total, 0)
        n_tot = await mpc.output(n_Total, 0)
        t_tot = await mpc.output(tot_res, 0)
    else:
        r_tot = 0
        s_tot = 0
        f_tot = 0
        p_tot = 0
        n_tot = 0
        t_tot = 0
    if DEBUG:
        print("SMPC 4 Done")
    """
    SMPC 5: Calculate Balances
    Balance
    """
    if OP == 2:
        da_bal = [secfxp(0.0)] * nodes

        for n in range(nodes):
            da = mpc.sum(mpc.schur_prod(D[n], N[n]))
            da_bal[n] = da

        da_out = await mpc.output(da_bal, 0)
        if DEBUG:
            print("SMPC 5 Output DA")

        flex = [secfxp(0.0)] * num_time
        if MODE in [1, 2, 3]:
            for t in range(num_time):
                flex_full = mpc.scalar_mul(F[t], mpc.vector_sub(A_T[t], S[t]))
                flex[t] = flex_full

            flex_bal = n_vector_add(flex)
        elif MODE == 4:
            raise NotImplemented
        else:
            flex_bal = [secfxp(0.0)] * nodes

        flex_out = await mpc.output(flex_bal, 0)
        if DEBUG:
            print("SMPC 5 Output Flex")

        if mpc.pid == 0:
            zB = [a + b for a, b in zip(da_out, flex_out)]
        else:
            zB = 0

        if DEBUG:
            print("SMPC 5 Output Full")
    else:
        zB = 0
    if DEBUG:
        print("SMPC 5 Done")
    """
    Output
    """
    if mpc.pid == 0:
        result_dict = {"zC": gen_rounding(zC, decimals),
                       "zP": gen_rounding(zP, decimals),
                       "zQ": gen_rounding(zQ, decimals),
                       "zU": gen_rounding(zU, decimals),
                       "zA": gen_rounding(zA, decimals),
                       "zRV": gen_rounding(zRV, decimals),
                       "zRF": gen_rounding(zRF, decimals),
                       "zD": gen_rounding(zD, decimals),
                       "zF": gen_rounding(zF, decimals),
                       "r_tot": gen_rounding(r_tot, decimals),
                       "s_tot": gen_rounding(s_tot, decimals),
                       "f_tot": gen_rounding(f_tot, decimals),
                       "p_tot": gen_rounding(p_tot, decimals),
                       "n_tot": gen_rounding(n_tot, decimals),
                       "t_tot": gen_rounding(t_tot, decimals),
                       "zB": gen_rounding(zB, decimals)}

        result_string = json.dumps(result_dict)
        with open(res_globals, "w") as outfile:
            outfile.write(result_string)

    await mpc.shutdown()  # disconnect, but only once all other parties reached this point


if __name__ == '__main__':
    mpc.run(main())
