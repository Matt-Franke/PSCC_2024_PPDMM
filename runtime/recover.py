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

# from modules import trivia

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
res_globals = os.path.normpath("./runtime/temp/recov.json")


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


def bfs(outflows, root, nodes):
    """
    BFS, returning for each node the number of hops to root and its ancestor
    :param outflows:
    :param root:
    :return:
    """
    dist = [0 for _ in range(nodes)]  # how many hops away from root
    anc = [root for _ in range(nodes)]  # ancestor (start with root)
    queue = [root]
    visited = []

    while queue:
        curr = queue.pop(0)
        visited.append(curr)

        for flows in outflows[curr]:
            t = flows[1]  # where to
            if t not in visited:  # so we dont go backwards
                anc[t] = curr
                dist[t] = dist[curr] + 1
                queue.append(t)

    return anc, dist


class UnionFind:
    """
    Basically a DisJoint set impl.
    We can find the parent for elements, do unions and return all elements with same parent (minus the parent)
    We do some path compression by having finds unioning until our parent is our grandparents parent
    We also do rank optimization, where rank here is distance to root

    Based in parts on "https://python.plainenglish.io/union-find-data-structure-in-python-8e55369e2a4f"
    """

    def __init__(self, dist, nodes, liars):
        """
        For the nodes, we have their direct ancestor and also the distance to root
        :param ancestor:
        :param dist:
        :param nodes:
        """
        self.ancestor = [n for n in range(nodes)]  # since we only want liars to be grouped!!!
        self.dist = dist
        self.nodes = nodes
        self.liars = liars

    def find(self, a):
        """
        Find our ancestor with path compression
        :param a: us
        :return: our ancestor
        """
        while a != self.ancestor[a]:
            self.ancestor[a] = self.ancestor[self.ancestor[a]]  # up to our grandfather
            a = self.ancestor[a]

        return a

    def union(self, a, b):
        """
        Union with rank opt the two elements
        This means we give whichever one is further away from root the ancestor of the other
        :param a: first element
        :param b: second element
        :return: boolean about whether anything changed
        """
        change = False

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            change = True
            if self.dist[a] > self.dist[b]:
                self.ancestor[a] = root_b
            else:  # so we err for a in a tie
                self.ancestor[b] = root_a

        return change

    def check(self, a, b):
        """
        Are they in the same set?
        :param a:
        :param b:
        :return:
        """
        return self.find(a) == self.find(b)


def islands_gen(outflows, L):
    """
    For all unique ancestors, return a list of the nodes with the same parent (besides the parent itself)
    :return:
    """
    unique_a = set(L.ancestor)
    islands = []

    for a in unique_a:
        isle = [i for i in range(L.nodes) if L.find(i) == a and i != a]
        if not isle and a in L.liars:
            # Check if we are really not connected to any liars
            friends = [j[1] for j in outflows[a] if j[1] in L.liars]
            if not friends:
                isle = [a]
        if isle:
            islands.append(isle)

    return islands


def get_islands(outflows, root, known_nodes, unknown_nodes, nodes):
    """
    Liars that connect to each other directly or are siblings are bounded to each other.
    This function turns the

    :param outflows: adjacency list (both directions)
    :param root: root node
    :param known_nodes: list of nodes that are not lying
    :param unknown_nodes: list of nodes that are lying
    :param nodes: total number of nodes
    :return:
    """
    # Do a BFS, determining for all their ancestor (a) and their rank
    anc, dist = bfs(outflows, root, nodes)
    # Islands are basically the result of union-finds of the liars:
    # We thus connect liars with shared ancestors together
    # This way clumps will be all under their shared ancestor (i.e node closest to root)
    # And lying siblings with an honest ancestor are also together!
    L = UnionFind(dist, nodes, unknown_nodes)

    for j in unknown_nodes:
        L.union(j, anc[j])  # Connect ourselves to our ancestor

    for j in unknown_nodes:
        # to make sure the path is compressed
        L.find(j)

    islands = islands_gen(outflows, L)  # this gets us a list of disjoint sets that are all liars with the same parent!

    # islands are made top down

    # Sanity check: no known nodes in any islands
    liar_islands = []
    for isle in islands:
        check = sum([1 for i in isle if i in known_nodes])
        if not check:
            liar_islands.append(isle)

    return islands, dist


async def main():
    """
    This is our MPC code
    :return:
    """
    await mpc.start()  # connect to all other parties

    print("Executing with {} nodes with threshold {}".format(nodes, mpc.threshold))

    #######################
    ##      SETUP        ##
    #######################
    with open(os.path.normpath("./runtime/temp/parameters.json"), 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)

    up_set = json_object["up_set"]
    dw_set = json_object["dw_set"]

    root = 0
    anc = [u[1] if len(u) > 1 else root for u in up_set]
    chi = [d[1:] for d in dw_set]

    """
    What outflows is supposed to look like:
    """
    outflows = [[0]] * len(up_set)

    for i in range(len(up_set)):
        neighbors = [anc[i]] + chi[i]
        oo = [[i, n] for n in neighbors]
        outflows[i] = oo

    path_state = os.path.normpath("./runtime/states/" + str(mpc.pid) + ".json")
    with open(path_state, 'r') as openfile:
        # Reading from json file
        input_dict = json.load(openfile)

    # CHANGE THE INPUT
    p = input_dict["ni_act"]  # only our own
    P = input_dict["fP_act"]  # only our own
    num_time = input_dict["num_time"]

    # GET LIARS FROM INPUT AS WELL!
    with open(os.path.normpath("./runtime/temp/iff.json"), 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)

    honest = json_object["honest"]
    liars = json_object["liars"]

    # Step 1: Input values
    if mpc.pid in liars:
        # Default sharing
        P_sec = [secfxp(0)] * num_time
        p_sec = [secfxp(0)] * num_time
    else:
        P_sec = list(map(secfxp, P))
        p_sec = list(map(secfxp, p))

    line_pow = mpc.input(P_sec)
    node_pow = mpc.input(p_sec)

    # Step 3: Gather islands based on topology and list of liars
    islands, dist = get_islands(outflows, root, honest, liars, len(up_set))

    if DEBUG:
        print("Node {}: Islands found: {}".format(mpc.pid, islands))

    # Step 4: Implicit since we have passive security
    # pass

    for isle in islands:
        n = len(isle)  # number of nodes in this island

        # Step 5: Net injections of each islands
        # Given balance, what is injected is equal to what flows minus what flows in
        Ancestor_isle = list(set(anc[j] for j in isle if anc[j] not in isle))

        Children_isle = []
        for jj in isle:
            cc = chi[jj]
            for ii in cc:
                if ii not in isle:
                    Children_isle.append(ii)
        Children_isle = list(set(Children_isle))

        if DEBUG:
            print("Node {}: For isle {}, A: {}, C:{}".format(mpc.pid, isle, Ancestor_isle, Children_isle))

        zero = [secfxp(0.0)] * num_time
        aas = [zero]
        ccs = [zero]
        for a in Ancestor_isle:
            aas.append(mpc.vector_add(line_pow[a], node_pow[a]))
            for c in chi[a]:
                if c not in isle:
                    ccs.append(line_pow[c])

        if DEBUG:
            print("Node {}: aa:{}, cc:{}".format(mpc.pid, len(aas), len(ccs)))

        power_in_anc = n_vector_add(aas)
        power_in_chi = n_vector_add(ccs)
        power_i = mpc.vector_sub(power_in_anc, power_in_chi)
        kk = [line_pow[c] for c in Children_isle] + [zero]

        power_o = n_vector_add(kk)
        power_net = mpc.vector_sub(power_o, power_i)

        # Step 6: Individual shares of net injecton
        power_share = mpc.scalar_mul(secfxp(len(isle)), power_net)

        for i in isle:
            node_pow[i] = power_share

    node_out = [0] * nodes
    for i in range(nodes):
        node_out[i] = await mpc.output(node_pow[i], 0)

    """
    Output
    """
    if mpc.pid == 0:
        result_dict = {"p": node_out}

        result_string = json.dumps(result_dict)
        with open(res_globals, "w") as outfile:
            outfile.write(result_string)

    await mpc.shutdown()  # disconnect, but only once all other parties reached this point


if __name__ == '__main__':
    mpc.run(main())
