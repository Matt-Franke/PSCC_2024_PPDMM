"""
The DC-OPF Problem to get an error scenario
"""
import random

from modules import inputs, trivia
import time
import cvxpy as cp
import numpy
from scipy.linalg import sqrtm
import math
DEBUG = False

def generate_errrors(kids, n, dems, sd):
    """
    Given the tree structure of a network, create error terms for the net injections at each node
    and the flows between nodes
    We generates the errors for a few so-called source nodes and then generate the rest so
    the sum of all nodal errors is 0 and at each node the sum of incoming errors is that of outgoing errors
    We minimize the two via a quadratic OF, i.e. via SOC and using ECOS since it only lower bounds
    the OF term and gets in less numerical troubles
    """
    es = []
    nodal_n = [0.0 for _ in range(n)]  # nodal errors
    nodal_l = [0.0 for _ in range(n)]  # line errors, here the one of the feeder will just stay 0.0

    Cons = trivia.ConstraintFactory(0)
    tic = time.time()

    """
    Setup Costs
    """
    quad_nodes = [abs(1.0 + numpy.random.normal(0, scale=0.3)) for _ in range(n)]
    C_n = numpy.array([[quad_nodes[i] if i == j and i else 0 for i in range(n)] for j in range(n)])
    # noinspection PyTypeChecker
    C_n_sqrt: numpy.ndarray = sqrtm(C_n)

    quad_lines = [abs(1.0 + numpy.random.normal(0, scale=0.3)) for _ in range(n)]
    quad_lines[0] = 0.0
    C_f = numpy.array([[quad_lines[i] if i == j else 0 for i in range(n)] for j in range(n)])
    # noinspection PyTypeChecker
    C_f_sqrt: numpy.ndarray = sqrtm(C_f)

    """
    Setup Variables
    """
    e_N = cp.Variable(n, name="e_N")
    e_F = cp.Variable(n, name="e_F")
    t_N = cp.Variable(name="t_N")
    t_F = cp.Variable(name="t_F")

    """
    Setup Constraints
    """

    # Global Balance
    # sum of all nodal errors is 0
    Cons.create(sum(e_N) == 0.0, "Global Balance")

    # Nodal Balance
    # For each node, the error of the associated flow
    # is equal to sum of the error of the node and the sum of the flows to the children
    for i in range(n):
        if not i:
            # Feeder has no actual ancestor
            bal = 0.0 == e_N[i] + sum([e_F[j] for j in kids[i]])
            Cons.create(bal, "Balance Node " + str(i))
        else:
            # everyone else does have ancestors
            bal = e_F[i] == e_N[i] + sum([e_F[j] for j in kids[i]])
            Cons.create(bal, "Balance Node " + str(i))

    """
    Generate Source Nodes and Values
    """
    NS = [i for i in range(n)]
    splitter = random.randint(1, int(n/2))
    sources = random.sample(NS, splitter)  # We get measurements from everyone else
    sources.sort()
    for i in sources:
        gen_err = numpy.random.normal(0, scale=sd * float(abs(dems)))
        Cons.create(e_N[i] == gen_err, "Fix Node " + str(i)) # Fix them to generated error

    """
    Setup Objective Function
    """
    # Nodal
    quad_n = C_n_sqrt @ e_N
    Cons.create(cp.SOC(t_N, quad_n), "Quad Cost Nodes")

    # Flows
    quad_f = C_f_sqrt @ e_F
    Cons.create(cp.SOC(t_F, quad_f), "Quad Cost Flows")

    of = t_N + t_F

    """
    Solve!
    """
    obj = cp.Minimize(of)
    prob = cp.Problem(obj, Cons.getAll())

    toc = time.time()
    if DEBUG:
        print(f"Took {toc - tic} seconds to Create Problem")

    tic = time.time()
    prob.solve(solver=cp.ECOS, ignore_dpp=True)
    toc = time.time()
    if DEBUG:
        print(f"Took {toc - tic} seconds to Solve Problem")
        print("status:", prob.status)
        print("optimal value", prob.value)

    """
    Recover the error terms and return them appropriately
    """
    for i in range(n):
        nodal_n[i] = e_N[i].value
        if i:
            nodal_l[i] = e_F[i].value

    for i in range(n):
        err = {"P": nodal_l[i], "p": nodal_n[i]}
        es.append(err)

    return es


if __name__ == "__main__":
    joint_df, tariff = inputs.retrieve_basic_inputs()
    days = 1
    incidence, resistance, reactance, susceptance, limits, nodes, lines, pf, der_cost = inputs.get_basic_network()
    connect, outflows = inputs.generate_connect_outflows(nodes, lines, incidence)
    ancestor, children, tree_order = trivia.get_ancestry(nodes, outflows, 0)
    dem_forecast = joint_df["Demand (MW)"].values[:24 * days]
    rel_sd = 0.1
    for t in range(24 * days):
        curr_demands = dem_forecast[t]
        errors = generate_errrors(children, nodes, curr_demands, rel_sd)
        print(errors)

