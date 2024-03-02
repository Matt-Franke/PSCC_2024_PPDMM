"""
This MPC protocol is for the normal operation each iteration
TOTAL OF, new globals, termination decision
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
TEST = True
res_globals = os.path.normpath("./runtime/temp/cov.json")


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


async def main():
    """
    This is our MPC code
    :return:
    """
    await mpc.start()  # connect to all other parties

    print("Executing with {} nodes with threshold {}".format(nodes, mpc.threshold))

    """
    INPUT
    """
    path_state = os.path.normpath("./runtime/temp/cov_" + str(mpc.pid) + ".json")
    with open(path_state, 'r') as openfile:
        # Reading from json file
        input_dict = json.load(openfile)

    vals = numpy.array(input_dict["vals"])  # single vector of length num_time
    num_time = input_dict["num_time"]

    if DEBUG:
        print("Node: {}, Answer:{}".format(mpc.pid, vals))

    f_t = await mpc.transfer(vals)

    """
    Covariance calculation
    """
    # M = secfxp.array(numpy.zeros((nodes, num_time)))
    N = num_time
    # Let f_act be the list of arrays of actual forecast errors
    # This did not work, so let us just use the stacked one lmao
    if DEBUG:
        print("Calculate the initial input")

    M = secfxp.array(numpy.stack(f_t))  # a N*T matrix
    # Tranpose M so that the rows are the timesteps
    if DEBUG:
        print("Calculate the transpose")
    M.integral = False  # we know that all values are between 0 and 1.0 (and specifically exclude 1.0)
    M = mpc.np_transpose(M)  # T * N matrix

    # Scale M by 1/(N-1) to calculate x'
    if DEBUG:
        print("Calculate the scaled input")
    scaler = 1 / (N - 1)
    M = mpc.np_multiply(M, secfxp(scaler))  # this is what breaks it lol

    # Scale by 1/N to calculate averages of x' and then sum them up
    if DEBUG:
        print("Calculate the averages")
    A = secfxp(1 / N) * M
    A = mpc.np_sum(A, axis=0)  # row of length N

    # Get Diff by subtracting A from each row of the matrix
    if DEBUG:
        print("Calculate the diff")
    D = mpc.np_copy(M)
    for t in range(num_time):
        diff = D[t] - A
        D = mpc.np_update(D, t, diff)

    # Get Sigma by doing the outer product of each row with itself
    # and then adding them all together
    if DEBUG:
        print("Calculate the outer products")
    S = secfxp.array(numpy.zeros((nodes, nodes)))  # N times N matrix

    for t in range(num_time):
        k = mpc.np_outer(D[t], D[t])
        S += k

    # S is at this point a matrix of variances (i.e. squared values)
    # to get s, we just add all the elements of S and take the sqrt
    s2 = mpc.np_sum(S)
    s = f_sqrt(s2)

    S_out = await mpc.output(S)
    s_out = await mpc.output(s)
    if DEBUG:
        print("Node {}, s: {}".format(mpc.pid, s_out))
        print(S_out)

    """
    Output
    """
    if mpc.pid == 0:
        result_dict = {"S": S_out.tolist(),
                       "s": s_out}

        result_string = json.dumps(result_dict)
        with open(res_globals, "w") as outfile:
            outfile.write(result_string)

    await mpc.shutdown()  # disconnect, but only once all other parties reached this point


if __name__ == '__main__':
    mpc.run(main())
