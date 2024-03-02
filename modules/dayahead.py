"""
Here the joint DA Energy+Reserve with Affine Decs market is realised
"""
import math
import random

import cvxpy as cp
import time
from math import sqrt
import numpy
import pandas
from scipy.linalg import sqrtm
from scipy.stats import norm
from modules import scenario, trivia, error_flow, update2reality


# noinspection PyAttributeOutsideInit
class Auction:
    """
        The Optimization Problem for the Day Auction that is fully deterministic, i.e. no uncertainty
        """

    def __init__(self, C: scenario.Community):
        """
        Import and derive the datasets required for the Problem
        """
        self.num_time = C.num_time
        self.nodes = C.nodes
        self.lines = C.lines
        self.C = C
        self.CC = C.CC
        self.setup()

    def setup(self):
        """
        Get the basic constants for the problem
        :param k:
        :param cost:
        :return:
        """
        ##########################
        # LOAD PARAMETERS
        ##########################
        """
        Voltage Parameters
        """
        self.u_root = 1.0  # voltage magnitude of C.feeder
        self.u_min = cp.Parameter(value=0.95 ** 2, name="u_min")
        self.u_max = cp.Parameter(value=1.05 ** 2, name="u_max")

        """
        Cost Parameters
        """
        self.util = cp.Parameter((self.nodes,), value=self.C.utility, name="util")
        self.l_cost = cp.Parameter((self.nodes,), value=self.C.lin_cost, name="lin_cost")
        self.q_cost = cp.Parameter((self.nodes,), value=self.C.quad_cost, name="quad_cost")

        """
        Line Parameters
        """
        # See "Distributed Generation Hosting Capacity Evaluation for Distribution Systems Considering the Robust Optimal Operation of OLTC and SVC"
        # Designed for approximation of flows in MW/MVA so better not touch it and change the rest!
        self.a1 = [1, 1, 0.2679, -0.2679, -1, -1, -1, -1, -0.2679, 0.2679, 1, 1]
        self.a2 = [0.2679, 1, 1, 1, 1, 0.2679, -0.2679, -1, -1, -1, -1, -0.2679]
        self.a3 = [-1, -1.366, -1, -1, -1.366, -1, -1, -1.366, -1, -1, -1.366, -1]

        self.circle_approx = len(self.a1)  # how many basically

        self.resit = cp.Parameter((self.nodes,), value=self.C.r, name="R")
        self.react = cp.Parameter((self.nodes,), value=self.C.x, name="X")
        self.fLim = cp.Parameter((self.nodes,), value=self.C.S_max, name="f_max")

        """
        Gen. Parameters
        """
        self.minZero = cp.Parameter(value=0.0, name="minZero")

        """
        Matrices for SOC constraints as per DLMP paper
        """
        # A[i][j] = 1 if edge i is part of path from the root node to node j
        A_list_dlmp = [[0 for _ in range(self.lines)] for _ in range(self.lines)]

        for ppp in self.C.parties:
            node_2_root = ppp
            while node_2_root != self.C.feeder:
                li = self.C.connect[node_2_root][
                    self.C.ancestor[node_2_root]]  # the line between the next node and its ancestor
                A_list_dlmp[li][ppp - 1] = 1
                node_2_root = self.C.ancestor[node_2_root]  # go one up

        self.A_dlmp = numpy.array(A_list_dlmp)
        self.A_inv_dlmp = numpy.linalg.inv(self.A_dlmp)
        A_dt_dlmp = self.A_dlmp.transpose()
        # Diagonal matrix of resistances
        Rd_dlmp = numpy.diag(numpy.array(self.C.r[1:]))
        # Basically the resistances on the path from root to the node
        self.R_dlmp = A_dt_dlmp * Rd_dlmp * self.A_dlmp
        # The inverse of the above
        self.R_inv_dlmp = numpy.linalg.inv(self.R_dlmp)
        C_g = numpy.array(
            [[self.C.quad_cost[i] if i == j and i else 0 for i in range(self.nodes)] for j in range(self.nodes)])
        # noinspection PyTypeChecker
        self.C_g_sqrt: numpy.ndarray = sqrtm(C_g)

        if self.CC:
            C_alpha = numpy.array(
                [[self.C.quad_cost[i] if i == j else 0 for i in range(self.nodes)] for j in range(self.nodes)])
            # noinspection PyTypeChecker
            self.C_alpha_sqrt: numpy.ndarray = sqrtm(C_alpha)

        """
        Chance Constraining!!!

        We assume that the demand have uncertainty

        For each, let their std. dev. sigma be 20% of their value.
        (since demand the same everywhere this makes things easier for now)

        Then let Sigma  be the list of the variances (i.e. std devs squared)
        Then eSigmae (in the original) is basically the sum of all the variances
        Which is a fixed value (i.e. a parameter!) that is >= 0
        Let this value be s_2. We want the square rootof this which we will call s.
        We also need the quantile function for which we can just use norm.ppf(1 - epsilon)
        Where the relevant epislon is also a fixed value, we will use 0.05 for this
        let z = norm.ppf(1 - epsilon)

        We also have the sum of all alphas is 1.0
        All alphas are between 0 and 1
        """
        if self.CC:
            # Full CC, Global Balance
            e_g = 0.05
            self.z_g = cp.Parameter(value=norm.ppf(1 - e_g), name="Risk Term G")

            e_v = 0.01
            # Since we use 2*z_v through, and it will get mad at us otherwise
            self.z_v = cp.Parameter(value=2.0 * norm.ppf(1 - e_v), name="Risk Term V")

            e_f = 0.01
            self.z_f = cp.Parameter(value=norm.ppf(1 - e_f), name="Risk Term F")

    # noinspection PyDictCreation
    def run_simulation(self, BC: bool):
        """
        Here the problem is declared, executed and the results written into a dataframe
        Parameters:
        # number of days, whether to include network constraints
        St Devs for the loads, pv and wind gen and the spot price
        This allows us to fuzz the values of each node (around the mean of the given value)

        """
        begint = time.time()
        # Names of Variables
        num_time = self.num_time
        print("Running with numtime of {}".format(num_time))
        # So access the variable gP_1 at timestep 2 by gP[2][1]
        gP = cp.Variable((self.nodes, num_time), name="GenP")
        gQ = cp.Variable((self.nodes, num_time), name="GenQ")
        fP = cp.Variable((self.nodes, num_time), name="FP")
        fQ = cp.Variable((self.nodes, num_time), name="FQ")
        us = cp.Variable((self.nodes, num_time), name="U")

        B = cp.Variable((self.nodes, num_time), name="Bat_SOC")  # Battery State of Charge

        s_P = cp.Variable(num_time, name="s_P")
        s_Q = cp.Variable(num_time, name="s_Q")
        l_P = cp.Variable(num_time, name="l_P")
        l_Q = cp.Variable(num_time, name="l_Q")

        quad_genP = cp.Variable(num_time, name="quad_genP")

        # self.CC VARIABLES
        if self.CC:
            alpha = cp.Variable((self.nodes, num_time), name="Alpha")
            quad_alpha = cp.Variable(num_time, name="quad_alpha")
            tv = cp.Variable((self.nodes, num_time), name="t_V")
            tf = cp.Variable((self.nodes, num_time), name="t_F")
            rho_v = cp.Variable((self.nodes, num_time), name="rho_v")
            rho_f = cp.Variable((self.nodes, num_time), name="rho_f")

        """
        RESULT MATRIX SETUP
        """
        lambda_0 = []
        shadow_prices = [[] for _ in range(self.nodes)]
        result_g_P = [[] for _ in range(self.nodes)]
        result_g_Q = [[] for _ in range(self.nodes)]
        result_dP = [[] for _ in range(self.nodes)]
        demand = [[] for _ in range(self.nodes)]
        result_SOC = [[] for _ in range(self.nodes)]
        result_f_P = [[] for _ in range(self.nodes)]
        result_f_Q = [[] for _ in range(self.nodes)]
        result_l_P = []
        result_l_Q = []
        result_s_P = []
        result_s_Q = []
        samePrice = []
        result_t_P = []
        result_t_Q = []
        result_d_P = []
        result_d_Q = []

        der_node = []
        for i in range(self.nodes):
            if i in self.C.pv:
                DER = abs(self.C.forecast["PV"])
            elif i == self.C.wind[0]:
                DER = abs(self.C.forecast["Wind1"])
            elif i == self.C.wind[1]:
                DER = abs(self.C.forecast["Wind2"])
            else:
                DER = [0.0 for _ in range(num_time)]  # by default no renewable
            der_node.append(DER)

        dem_node = self.C.forecast["dP"][:]

        der_array = numpy.array(der_node)
        dP_array = numpy.array(dem_node)
        total_demand = numpy.sum(dP_array, axis=1)
        # All battery discharge at limit + DERs with no load
        max_P_outflow = [sum([der_node[i][t] + self.C.g_P_max[i] for i in range(self.nodes)]) for t in range(num_time)]

        # No reactive generation in the network!!
        max_Q_outflow = [0.0 for _ in range(num_time)]

        # All batteries charge at limit + Full P Demand + No DER
        max_P_inflow = max_P_inflow = [sum([-1.0 * self.C.g_P_min[i] + dem_node[t][i] for i in range(self.nodes)]) for t
                                       in range(num_time)]

        # Full Q Demand
        max_Q_inflow = [sum([1.0 * self.C.pf * dem_node[t][i] for i in range(self.nodes)]) for t in range(num_time)]
        u_root = 1.0

        maxes = {"P_I": max_P_inflow, "P_O": max_P_outflow, "Q_I": max_Q_inflow, "Q_O": max_Q_outflow, "U0": u_root}

        if self.CC:
            result_alpha = [[] for _ in range(self.nodes)]
            alpha_price = []
            result_st_dev = [[] for _ in range(self.num_time)]

        balances = [[] for _ in range(self.nodes)]  # We know they all start with 0!
        surplus = []

        if self.CC:
            flag = 0
        flow_surplus = [0.0 for _ in range(self.num_time)]

        social_welfare = self.minZero
        Cons = trivia.ConstraintFactory(0)

        tic = time.time()
        # For each timestep t:
        for t in range(self.num_time):
            print("Timestep " + str(t))

            """
            Update Constants
            """
            if self.CC:
                result_st_dev[t] = self.C.forecast["SD"][t]
                var_all = [i ** 2 for i in self.C.forecast["SD"][t]]
                ss_2 = sum(var_all)
                ss = sqrt(ss_2)
                S_dlmp = numpy.diag(numpy.array(var_all[1:]))  # Sigma
                S_inv_dlmp = sqrtm(S_dlmp)  # Sigma ^(1/2)

                s = cp.Parameter(value=ss, name="Root of Sum of Vars")

            # Wind has 0 cost spillage, so we can use it as a flexible generator
            # At the Day ahead, it can provide at most its forecast
            # Thus, it may be asked to perform less than that for the DA, but it can just spill whatever is extra
            # This also allows it to provide down regulation (for cost of 0) and up-regulation by spilling less!
            # Batteries at all nodes with PV are the actual flexible generation at those nodes!
            if self.C.Q_batt:
                # As per Bazrafshan2017
                # Decentralized Stochastic Optimal Power Flow in Radial Networks with Distributed Generation
                for gg in self.C.gens:
                    Q_m = sqrt(self.C.max_ap ** 2 + self.C.forecast["PV"][gg] ** 2)
                    # Wind and Batteries cannot generate reactive energy!
                    self.C.g_Q_max[gg] = Q_m
                    self.C.g_Q_min[gg] = -1.0 * Q_m

            """
            Create Problem
            """
            """
            VARIABLES (for now, until I figure out how to reset just the constraints)
            """
            # FEEDER
            Cons.create(self.minZero <= s_P[t], "min SP" + " at time " + str(t))
            Cons.create(self.minZero <= s_Q[t], "min SQ" + " at time " + str(t))
            Cons.create(self.minZero <= l_P[t], "min LP" + " at time " + str(t))
            Cons.create(self.minZero <= l_Q[t], "min LQ" + " at time " + str(t))

            Cons.create(s_P[t] <= 1.0 * abs(maxes["P_O"][t]), "P Gen Min Node " + " at time " + str(t))
            Cons.create(l_P[t] <= 1.0 * abs(maxes["P_I"][t]), "P Gen Max Node " + " at time " + str(t))
            Cons.create(s_Q[t] <= 1.0 * abs(maxes["Q_O"][t]), "Q Gen Min Node " + " at time " + str(t))
            Cons.create(l_Q[t] <= 1.0 * abs(maxes["Q_I"][t]), "Q Gen Max Node " + " at time " + str(t))

            # feeder has no inflow and no ancestor
            Cons.create(fP[0, t] == self.minZero, "P Flow at Feeder at time " + str(t))
            Cons.create(fQ[0, t] == self.minZero, "Q Flow at Feeder at time " + str(t))

            if self.CC:
                Cons.create(self.minZero == rho_v[0, t],
                            "Wired rho_v " + str(t))  # since feeder does not have V or pF/pQ
                Cons.create(self.minZero == rho_f[0, t],
                            "Wired rho_f " + str(t))  # since feeder does not have V or pF/pQ

            # gP and gQ are aux for the flows
            Cons.create(gP[self.C.feeder, t] == l_P[t] - s_P[t], "P Gen at Feeder at time " + str(t))
            Cons.create(gQ[self.C.feeder, t] == l_Q[t] - s_Q[t], "Q Gen at Feeder at time " + str(t))

            # FULL CC OR NOT
            if self.CC:
                # Constraint on tv and tf
                for i in self.C.parties:
                    Cons.create(self.minZero <= tv[i, t], "min t_v" + str(i) + " at time " + str(t))
                    Cons.create(self.minZero <= tf[i, t], "min t_f" + str(i) + " at time " + str(t))

                # Const 6: Voltage Limits for all i in P
                # u_min + 2z_v*t_v <= u_i <= u_max - 2z_v*t_v
                # Where u_min = (v_i_min)^2 and u_max = (v_i_max)^2
                # We use the same limits for all namely 0.95 to 1.05
                for i in self.C.parties:
                    Cons.create(us[i, t] >= self.u_min + self.z_v * tv[i, t],
                                "Min U" + str(i) + " at time " + str(t))
                    Cons.create(us[i, t] <= self.u_max - self.z_v * tv[i, t],
                                "Max U" + str(i) + " at time " + str(t))

                # C12: Flow limits (with gP, just set to 0 if there is none)
                # Now there is CC at the first term!
                # a_1_c*(f_P_l + z_f*t_f) + a_2_c*f_Q_i + a_3_c*S_max_i <= 0 , for c in range(12)
                for i in range(1, self.nodes):
                    for la in range(self.circle_approx):
                        lin_approx = self.a1[la] * (fP[i, t] + self.z_f * tf[i, t]) + self.a2[la] * fQ[i, t] + \
                                     self.a3[la] * self.fLim[i]
                        Cons.create(lin_approx <= 0, "LA+ " + str(la) + "for Node " + str(i) + " at time " + str(t))

                        lin_approx = self.a1[la] * (fP[i, t] - self.z_f * tf[i, t]) + self.a2[la] * fQ[i, t] + \
                                     self.a3[la] * self.fLim[i]
                        Cons.create(lin_approx <= 0, "LA- " + str(la) + "for Node " + str(i) + " at time " + str(t))

                # Sum constraint on alpha from u
                # \sum_{j \in P} R^*_{ij}\rho_j^v == alpha_i \forall i in G (i.e. those that offer flex)
                for i in self.C.parties:
                    # note since we exclude root, row 0 is actually for node 1 and so on
                    li = self.C.connect[i][self.C.ancestor[i]]
                    rrhou_LHS = sum(
                        [self.R_inv_dlmp[li][j] * rho_v[j, t] for j in range(self.lines)])  # j from 1 to n
                    Cons.create(rrhou_LHS == alpha[i, t], "Sum U on alpha " + str(i) + " at time " + str(t))

                # Sum constraint on alpha from f
                # \sum_{j \in P} A^*_{ij}\rho_j^f == alpha_i \forall i in G (i.e. those that offer flex)
                for i in self.C.parties:
                    li = self.C.connect[i][self.C.ancestor[i]]
                    rrhof_LHS = sum([self.A_inv_dlmp[li][j] * rho_f[j, t] for j in range(self.lines)])
                    Cons.create(rrhof_LHS == alpha[i, t], "Sum F on alpha " + str(i) + " at time " + str(t))

                # SOC constraint from u
                # x = S_inv_dlmp*(rho_v[i]*ones^T + R_dlmp[i])
                for i in self.C.parties:
                    li = self.C.connect[i][self.C.ancestor[i]]
                    e_dlmp = numpy.ones((self.lines,))
                    x_r = rho_v[i, t] * e_dlmp.transpose()
                    x_l = self.R_dlmp[li]
                    x = (x_l + x_r) @ S_inv_dlmp
                    Cons.create(cp.SOC(tv[i, t], x), "SOC for u " + str(li) + " at time " + str(t))

                # SOC constraint from f
                # x = S_inv_dlmp*(A_dlmp[i] - rho_f[i]*ones^T)
                # for i in range(1, nodes+1):
                for i in range(1, self.nodes):
                    li = self.C.connect[i][self.C.ancestor[i]]
                    e_dlmp = numpy.ones((self.lines,))
                    x_l = self.A_dlmp[li]
                    x_r = rho_f[i, t] * e_dlmp.transpose()
                    x = (x_l + x_r) @ S_inv_dlmp

                    Cons.create(cp.SOC(tf[i, t], x), "SOC for f " + str(li) + " at time " + str(t))
            else:
                # Const 6: Voltage Limits for all i in P
                # Where u_min = (v_i_min)^2 and u_max = (v_i_max)^2
                # We use the same limits for all namely 0.95 to 1.05
                for i in self.C.parties:
                    Cons.create(us[i, t] >= self.u_min, "Min U" + str(i) + " at time " + str(t))
                    Cons.create(us[i, t] <= self.u_max, "Max U" + str(i) + " at time " + str(t))

                # Cons 4B: Flow Limits (Linear Approx) for all i in P
                for i in self.C.parties:
                    for la in range(self.circle_approx):
                        lin_approx = self.a1[la] * fP[i, t] + self.a2[la] * fQ[i, t] + self.a3[la] * self.fLim[i]
                        Cons.create(lin_approx <= 0, "LA+ " + str(la) + "for Node " + str(i) + " at time " + str(t))

            # Cons 1: Power Balance between Feeder and LEM
            # (l-s) is the inflow into network
            # (l_P-s_P) = sum_{j in C_0}(f_P_j)
            # (l_Q-s_Q) = sum_{j in C_0}(f_Q_j)
            Cons.create(l_P[t] - s_P[t] == sum([fP[j, t] for j in self.C.children[self.C.feeder]]),
                        "P Balance at Feeder" + " at time " + str(t))
            Cons.create(l_Q[t] - s_Q[t] == sum([fQ[j, t] for j in self.C.children[self.C.feeder]]),
                        "Q Balance at Feeder" + " at time " + str(t))

            # Cons 2: Power Flow Calculation for all i in P (C.parties)
            # Can opt with more list comprehension but that is just too cryptic
            for i in self.C.parties:
                if i in self.C.pv:
                    DER = abs(self.C.forecast["PV"][t])
                elif i == self.C.wind[0]:
                    DER = abs(self.C.forecast["Wind1"][t])
                elif i == self.C.wind[1]:
                    DER = abs(self.C.forecast["Wind2"][t])
                else:
                    DER = 0.0  # by default no renewable
                # f_P_i + (g_P_i-d_P_i) = sum_{j in C_i}(f_P_j)
                fP_bal = fP[i, t] + gP[i, t] - self.C.forecast["dP"][t][i] + DER == sum(
                    [fP[j, t] for j in self.C.children[i]])
                Cons.create(fP_bal, "P Balance at Node " + str(i) + " at time " + str(t))

                # f_Q_i + (g_Q_i-d_Q_i) = sum_{j in C_i}(f_Q_j)
                fQ_bal = fQ[i, t] + gQ[i, t] - self.C.forecast["dQ"][t][i] == sum(
                    [fQ[j, t] for j in self.C.children[i]])
                Cons.create(fQ_bal, "Q Balance at Node " + str(i) + " at time " + str(t))

            # Cons 3: Voltage Calculation for all i in P
            for i in self.C.parties:
                # u_i = u_A_i - 2*(r_i*f_P_i + x_i*f_Q_i)
                u_drop = (us[i, t] == us[self.C.ancestor[i], t] - 2 * (
                        self.resit[i] * fP[i, t] + self.react[i] * fQ[i, t]))
                Cons.create(u_drop, "Voltage at Node " + str(i) + " at time " + str(t))

            # Const 3A: Fix u_root to 1
            Cons.create(us[self.C.feeder] == self.u_root, "Voltage at Feeder" + " at time " + str(t))

            if self.CC:
                # LIMITS ON ALPHA
                for i in range(self.nodes):
                    if i in self.C.gens or i == self.C.feeder:
                        # 0 <= alpha_i <= 1
                        Cons.create(alpha[i, t] >= 0.0, "Alpha Min Node " + str(i) + " at time " + str(t))
                        Cons.create(alpha[i, t] <= 1.0, "Alpha Max Node " + str(i) + " at time " + str(t))
                    else:
                        Cons.create(alpha[i, t] == 0.0, "Alpha Min Node " + str(i) + " at time " + str(t))

                # Sum of alphas is 1
                Cons.create(sum(alpha[:, t]) == 1.0, "Alpha Balance" + " at time " + str(t))

            # Const 5: Generation limits for all i in P
            # This stays the same between Ratha and Mieth: P CC, Q deterministic
            # z and s are calculated mostly the same, just now there is 3 z's
            # P gen is chance constrained
            if self.CC:
                for i in self.C.gens:
                    # With battery flows
                    # g_P_i_min <= g_P_i
                    Cons.create(gP[i, t] - self.z_g * alpha[i, t] * s >= self.C.g_P_min[i],
                                "P Gen Min Node " + str(i) + " at time " + str(t))
                    # g_P_i <= g_P_i_max
                    Cons.create(gP[i, t] + self.z_g * alpha[i, t] * s <= self.C.g_P_max[i],
                                "P Gen Max Node " + str(i) + " at time " + str(t))

                    if t == 0:
                        # C14: SOC at begin and end
                        Cons.create(B[i, t] == self.C.Q_midnight[i] * self.C.Q[i] - gP[i, t],
                                    "SOC @ Node " + str(i) + " at time " + str(t))
                    else:
                        # C13: SOC capture
                        Cons.create(B[i, t] == B[i, t - 1] - gP[i, t], "SOC @ Node " + str(i) + " at time " + str(t))

                    if t == (num_time - 1):
                        # 15: SOC at begin and end
                        Cons.create(B[i, t] - self.z_g * alpha[i, t] * s >= self.C.Q_midnight[i] * self.C.Q[i],
                                    "End SOC @ Node" + str(i))

                    # C16,17: SOC limits
                    Cons.create(B[i, t] - self.z_g * alpha[i, t] * s >= self.C.Q_min[i] * self.C.Q[i],
                                "SOC Min at time " + str(t))
                    Cons.create(B[i, t] + self.z_g * alpha[i, t] * s <= self.C.Q_max[i] * self.C.Q[i],
                                "SOC Max at time " + str(t))
            else:
                for i in self.C.gens:
                    # With battery flows
                    # g_P_i_min <= g_P_i
                    Cons.create(gP[i, t] >= self.C.g_P_min[i],
                                "P Gen Min Node " + str(i) + " at time " + str(t))
                    # g_P_i <= g_P_i_max
                    Cons.create(gP[i, t] <= self.C.g_P_max[i],
                                "P Gen Max Node " + str(i) + " at time " + str(t))

                    if t == 0:
                        # C14: SOC at begin and end
                        Cons.create(B[i, t] == self.C.Q_midnight[i] * self.C.Q[i] - gP[i, t],
                                    "SOC @ Node " + str(i) + " at time " + str(t))
                    else:
                        # C13: SOC capture
                        Cons.create(B[i, t] == B[i, t - 1] - gP[i, t], "SOC @ Node " + str(i) + " at time " + str(t))

                    if t == (num_time - 1):
                        # 15: SOC at begin and end
                        Cons.create(B[i, t] >= self.C.Q_midnight[i] * self.C.Q[i],
                                    "End SOC @ Node" + str(i))

                    # C16,17: SOC limits
                    Cons.create(B[i, t] >= self.C.Q_min[i] * self.C.Q[i],
                                "SOC Min at time " + str(t))
                    Cons.create(B[i, t] <= self.C.Q_max[i] * self.C.Q[i],
                                "SOC Max at time " + str(t))

            # SAME FOR BOTH
            # No reactive generation!
            # g_Q_i_min <= g_Q_i
            for i in self.C.parties:
                Cons.create(gQ[i, t] == 0.0, "Q Gen Min Node " + str(i) + " at time " + str(t))
                if i not in self.C.gens:
                    # g_P_i_min <= g_P_i
                    Cons.create(gP[i, t] == self.minZero, "P Gen Min Node " + str(i) + " at time " + str(t))
                    Cons.create(B[i, t] == self.minZero, "P Gen Min Node " + str(i) + " at time " + str(t))

            """
            OBJECTIVE
            """
            # Linear cost is just for active power flow injections
            linear_cost = sum([self.C.lin_cost[i] * gP[i, t] for i in self.C.parties]) + l_P[t] * self.C.forecast["GT"][
                t] - s_P[t] * self.C.forecast["FT"][t]
            # -1 * util_parties is a fixed term!

            # We do the quadratic costs for alpha and for gP via auxillary variable
            # that are lower bounded by a SOC constraint with the actual quadratic calculation

            # Constraint g_P
            gp_t = gP[:, t]
            quad_genP_x = self.C_g_sqrt @ gp_t
            Cons.create(cp.SOC(quad_genP[t], quad_genP_x), "Quad Cost P Gen" + " at time " + str(t))

            # Constraint alpha
            if self.CC:
                quad_alpha_x_r = s * alpha[:, t]
                quad_alpha_x = self.C_alpha_sqrt @ quad_alpha_x_r
                Cons.create(cp.SOC(quad_alpha[t], quad_alpha_x), "Quad Cost Flex" + " at time " + str(t))

            # Full Objective
            if self.CC:
                social_welfare += linear_cost + quad_genP[t] + quad_alpha[t]
            else:
                social_welfare += linear_cost + quad_genP[t]

        obj = cp.Minimize(social_welfare)

        """
        Solve!
        """
        prob = cp.Problem(obj, Cons.getAll())

        toc = time.time()
        print(f"Took {toc - tic} seconds to Create Problem")

        tic = time.time()
        prob.solve(solver=cp.ECOS, verbose=True, ignore_dpp=True)
        toc = time.time()
        print(f"Took {toc - tic} seconds to Solve Problem")
        print("status:", prob.status)
        print("optimal value", prob.value)

        if BC:
            """
            Find Binding Line Constraints and Determine the Dual Variable for each line
            """
            line_duals = [[0.0 for _ in range(self.lines)] for _ in range(self.num_time)]

            for t in range(self.num_time):
                for i in self.C.parties:
                    rezzies = [0 for _ in range(self.circle_approx)]
                    for la in range(self.circle_approx):
                        ra = Cons.getCon("LA+ " + str(la) + "for Node " + str(i) + " at time " + str(t)).dual_value
                        rezzies[la] = ra

                    tot = sum(rezzies)
                    line_duals[t][i - 1] = tot

            la_dual = numpy.array(line_duals)
            print(str(la_dual.max(initial=0)))

            bal_duals = [[0.0 for _ in range(self.nodes)] for _ in range(self.num_time)]

            for t in range(self.num_time):
                bal_duals[t][self.C.feeder] = -1.0 * Cons.getCon(
                    "P Balance at Feeder" + " at time " + str(t)).dual_value

                for i in self.C.parties:
                    bal_duals[t][i] = -1.0 * Cons.getCon(
                        "P Balance at Node " + str(i) + " at time " + str(t)).dual_value

            bP_dual = numpy.array(bal_duals)
            print(str(bP_dual.max(initial=0)))

        """
        Update Settlements
        """

        for t in range(self.num_time):
            result_l_P.append(l_P.value[t])
            result_l_Q.append(l_Q.value[t])
            result_s_P.append(s_P.value[t])
            result_s_Q.append(s_Q.value[t])

            if self.CC:
                alpha_price.append(-1 * Cons.getCon("Alpha Balance" + " at time " + str(t)).dual_value)

            l_0 = -1 * Cons.getCon("P Balance at Feeder" + " at time " + str(t)).dual_value
            lambda_0.append(-1 * Cons.getCon("P Balance at Feeder" + " at time " + str(t)).dual_value)
            shadow_prices[self.C.feeder].append(l_0)
            if self.CC:
                result_alpha[self.C.feeder].append(alpha[self.C.feeder, t].value)
            max_diff = 0
            gp_tot = 0
            gq_tot = 0
            dp_tot = sum(self.C.forecast["dP"][t])
            dq_tot = sum(self.C.forecast["dQ"][t])
            ls = [l_0]
            flow_surplus[t] += (result_l_P[t] - result_s_P[t])
            result_g_P[self.C.feeder].append(result_l_P[t] - result_s_P[t])
            demand[self.C.feeder].append(0.0)

            for i in self.C.parties:
                result_g_P[i].append(gP.value[i, t])
                result_g_Q[i].append(gQ.value[i, t])
                if self.CC:
                    result_alpha[i].append(alpha.value[i, t])
                gp_tot += gP.value[i, t]
                gq_tot += gQ.value[i, t]
                result_dP[i].append(self.C.forecast["dP"][t][i])

                result_SOC[i].append(B.value[i, t] if i in self.C.gens else 0.0)

                if i in self.C.pv:
                    DER = abs(self.C.forecast["PV"][t])
                elif i == self.C.wind[0]:
                    DER = abs(self.C.forecast["Wind1"][t])
                elif i == self.C.wind[1]:
                    DER = abs(self.C.forecast["Wind2"][t])
                else:
                    DER = 0.0  # by default no renewable

                demand[i].append(DER - self.C.forecast["dP"][t][i])
                # noinspection PyTypeChecker
                flow_surplus[t] += result_g_P[i][t] + DER - self.C.forecast["dP"][t][i]

                sp = -1 * Cons.getCon("P Balance at Node " + str(i) + " at time " + str(t)).dual_value

                max_diff = max(max_diff, abs(l_0 - sp))

                shadow_prices[i].append(sp)
                ls.append(sp)
                result_f_P[i].append(fP.value[i, t])
                result_f_Q[i].append(fQ.value[i, t])

            samePrice.append(max_diff)
            result_t_P.append(gp_tot)
            result_t_Q.append(gq_tot)
            result_d_P.append(dp_tot)
            result_d_Q.append(dq_tot)

            # Verify whether LA approx holds
            # by taking the calculated f flows and checking whether the sum of the squares at each
            # node is smaller than the square
            for i in self.C.parties:
                check = fP.value[i, t] ** 2 + fQ.value[i, t] ** 2 - self.fLim[i].value ** 2
                if check > 0:
                    print("Flow Check Failed")
                    raise ValueError

            """
            SETTLEMENT
            """
            total = 0

            if self.CC:
                # Flex Sharing Idea Two: Actors causing uncertainty pay for it
                # Since this is day ahead, split according to day ahead forecast errors
                SD_total = sum(result_st_dev[t])

            curr_bal = 0.0
            for i in range(self.nodes):
                if t:
                    curr_bal = balances[i][t - 1]
                if i == self.C.feeder:  # Feeder
                    bal_up = shadow_prices[self.C.feeder][t] * (result_l_P[t] - result_s_P[t])
                    if self.CC:
                        bal_up += alpha_price[t] * result_alpha[i][t]
                else:  # Everyone Else
                    if i in self.C.pv:
                        DER = abs(self.C.forecast["PV"][t])
                    elif i == self.C.wind[0]:
                        DER = abs(self.C.forecast["Wind1"][t])
                    elif i == self.C.wind[1]:
                        DER = abs(self.C.forecast["Wind2"][t])
                    else:
                        DER = 0.0  # by default no renewable

                    bal_up = shadow_prices[i][t] * (result_g_P[i][t] - result_dP[i][t] + DER)
                    if self.CC:
                        # Directionless
                        bal_up += alpha_price[t] * result_alpha[i][t]
                        # Your share of total uncertainty times alpha price (which is also total payment)
                        bal_up -= alpha_price[t] * (result_st_dev[t][i] / SD_total)
                total += bal_up
                balances[i].append(curr_bal + bal_up)

            surplus.append(total)

        total_residual = numpy.array(flow_surplus)

        final_balance = [balances[i][-1] for i in range(self.nodes)]
        print("Final Balance")
        print(final_balance)
        final_surplus = sum(final_balance)
        print("Final Surplus")
        print(final_surplus)
        print("Surplus")
        print(surplus)
        print("Flow Surplus")
        print(flow_surplus)
        print("Total Flow Surplus")
        print(sum(flow_surplus))
        print("Energy Residual")
        print(total_residual)
        print("Total Relative Surplus")
        print(sum(total_residual) * 100 / sum(total_demand))

        fh_ts = pandas.date_range('11/1/2013', periods=self.num_time, freq='1H')  # generate the timestamps

        # noinspection PyDictCreation
        results_dict = {"Timestep": fh_ts, "P Feed In": result_l_P,
                        "P Feed Out": result_s_P,
                        "P Local Gen": result_t_P, "P Local Dem": result_d_P, "Q Feed In": result_l_Q,
                        "Q Feed Out": result_s_Q, "Q Local Gen": result_t_Q,
                        "Q Local Dem": result_d_Q,
                        "FT": self.C.forecast["FT"], "GT": self.C.forecast["GT"],
                        "Surplus": surplus, "Spot": self.C.forecast["WM"]}

        results_dict["Rel Surplus"] = total_residual.tolist()
        if self.CC:
            results_dict["PI"] = alpha_price

        for i in range(self.nodes):
            results_dict["Shadow Price " + str(i)] = shadow_prices[i]

        for i in self.C.gens:
            results_dict["B_min" + str(i)] = [self.C.Q_min[i] * self.C.Q[i] for _ in range(self.num_time)]
            results_dict["B" + str(i)] = result_SOC[i]
            results_dict["B_max" + str(i)] = [self.C.Q_max[i] * self.C.Q[i] for _ in range(self.num_time)]

            results_dict["gP" + str(i)] = result_g_P[i]

        for i in self.C.gens:
            results_dict["gQ" + str(i)] = result_g_Q[i]

        if self.CC:
            for i in range(self.nodes):
                results_dict["Alpha" + str(i)] = result_alpha[i]

        for i in self.C.parties:
            if i in self.C.pv:
                DER = self.C.forecast["PV"]
            elif i == self.C.wind[0]:
                DER = self.C.forecast["Wind1"]
            elif i == self.C.wind[1]:
                DER = self.C.forecast["Wind2"]
            else:
                DER = [0.0 for _ in range(num_time)]
            results_dict["DER" + str(i)] = DER
            results_dict["Demand" + str(i)] = demand[i]

        for i in range(self.nodes):
            results_dict["U" + str(i)] = us.value[i].tolist()

        for i in self.C.parties:
            results_dict["fP" + str(i)] = result_f_P[i]
            results_dict["fQ" + str(i)] = result_f_Q[i]

        for i in range(self.nodes):
            results_dict["Balance N" + str(i)] = balances[i]

        self.results_df = pandas.DataFrame(results_dict)
        self.results_df.set_index("Timestep")

        endt = time.time()
        print(f"Took {endt - begint} seconds to Solve Problem")
        res_stat = "{} its, {:.2f} minutes, {:.4f} %, {:.6} abs.".format(1, (endt - begint) / 60,
                                                                         sum(total_residual) * 100 / sum(total_demand),
                                                                         sum(flow_surplus))

        print(res_stat)

        schedules = result_g_P  # Fixed Generation!
        # demand  # Net Demand = PV to Load + Battery to Load - Fixed Load
        if not self.CC:
            result_alpha = [[0]]
            alpha_price = [0]
        return schedules, demand, shadow_prices, result_alpha, alpha_price, balances, res_stat

    def get_actuals(self, dem_forecast, sd):
        """
        Takes in forecasts and applies a tighter SD to determine the actual net active power
        :param dem_forecast:
        :param sd:
        :return:
        """
        # sd is the relative Standard Deviation
        # the actual sd is scaled by the forecasted demand itself!
        # thus it is sd*abs(dem_forecast) that goes into the random function
        # then it is dem_forecast + that error
        actErr = [[numpy.random.normal(0, scale=sd * abs(dem_forecast[t])) for _ in
                   range(self.nodes)] for t in range(self.num_time)]  # apply uncertain demand

        return actErr

    def update_flows(self, schedules, demands, alphas, delta_i, delta_T):
        """
        This function takes the OT forecasts of errors and updates everyone's
        p, q values. It then activates the reserves according to the policies,
        which again updates the p and q values.
        Finally, based on the p and q updates, we update everyones P and Q values
        We return the scheduled P generation for everyone

        NOTE: if not CC, then there are no errors and thus all the updates are 0.0
        Note:
        self.real_values is hidden and it represents the actual values
        Schedules and Demands are shared to the actors and represent their updated net demand and schedule
        """
        # Collect updates throughout
        updates = [[0.0 for _ in range(self.nodes)] for _ in range(self.num_time)]
        flow_errors = [[0.0 for _ in range(self.nodes)] for _ in range(self.num_time)]

        # Init. the known values with the results
        # real_values[timestep][node]["quantity"] is then used to query
        # Get Demand
        self.real_values = [[
            {"P": 0, "p": 0.0 if i == self.C.feeder else self.results_df["DER" + str(i)][t] - self.results_df["Demand" + str(i)][t], "Q": 0,
             "q": 0.0 if i == self.C.feeder else -1.0 * self.results_df["Demand" + str(i)][t] * self.C.pf} for i in
            range(self.nodes)]
            for t in range(self.num_time)]  # Init so the indexing works
        # Add Generation to get net injections plus Power Flows
        for t in range(self.num_time):
            for i in range(self.nodes):
                if i == self.C.feeder:
                    # root has no P, Q flow => P=Q=0 and u=1 by def
                    # So only p and q are unknown
                    # p = l_p - s_p
                    # q = l_q - s_q
                    # we overwrite the default here, which is 0.0
                    self.real_values[t][i]["p"] = self.results_df["P Feed In"][t] - self.results_df["P Feed Out"][t]
                    self.real_values[t][i]["q"] = self.results_df["Q Feed In"][t] - self.results_df["Q Feed Out"][t]
                else:
                    self.real_values[t][i]["P"] = self.results_df["fP" + str(i)][t]
                    self.real_values[t][i]["Q"] = self.results_df["fQ" + str(i)][t]
                    # we increment the value here, since its our net demand
                    self.real_values[t][i]["q"] += 0.0

                # Update known values and demands with the error
                # No capping applied here, so we are still identica at this point
                # We just add the error here
                self.real_values[t][i]["p"] += delta_i[t][i]
                # Q gets P error times pf
                # self.real_values[t][i]["q"] -= delta_i[t][i]*self.C.pf
                # since we subtract the following, we need to subtract the error here
                # so on net it becomes an addition!
                demands[i][t] += delta_i[t][i]

                if self.CC:
                    # Trigger Response
                    # Update schedule without capping (since this is what we are asking of the actor)
                    # Update known values with capping (since this is what is actually able to happen)
                    # We cap by forcing the value x to be between g_p_min and g_p_max
                    # Q gets P error times pf
                    # self.real_values[t][i]["q"] -= alphas[i][t] * delta_T[t] * self.C.pf
                    schedules[i][t] -= alphas[i][t] * delta_T[t]

                    # Note we cap only the generation within the limits!!
                    gP_sched = self.results_df["gP" + str(i)][t] if i in self.C.gens else 0.0
                    self.real_values[t][i]["p"] += max(min(self.C.g_P_max[i], gP_sched - alphas[i][t] * delta_T[t]),
                                                       self.C.g_P_min[i])

                    # Updates the updates tracker
                    updates[t][i] = delta_i[t][i] - alphas[i][t] * delta_T[t]
                else:
                    updates[t][i] = delta_i[t][i]
        # Update known values of power flow
        # Use update2reality
        order = update2reality.gen_order(self.C.ancestor, self.C.children, self.nodes)
        # flow_errors for P, updates for p
        for t in range(self.num_time):
            for i in order:
                # iterate over the nodes in the order, knowing deterministically it will work
                # p has already been updated by updates in the above section
                # calculate error for P_flow, which is the sum of the children minus the update
                flow_errors[t][i] = sum(flow_errors[t][j] for j in self.C.children[i]) - updates[t][i]
                if i == self.C.feeder:
                    # for the feeder there is no flow to update
                    # not sure how to handle this!!
                    self.real_values[t][i]["P"] += 0.0
                else:
                    # normal nodes have flows that are updates
                    self.real_values[t][i]["P"] += flow_errors[t][i]

        # Return the updated schedules and demands
        return schedules, demands

    def penalty(self, deltas_i, delta_total, up_reg, dw_reg, spot_ws):
        """
        Calculates imbalance penalties based on the 2 price scheme

        :param deltas_i: Individual deviation from scheduled for each party at each timestep
        :param delta_total: Total deviation at each timestep
        :param up_reg: Wholesale Up Balancing Price (if total Deviation negative)
        :param dw_reg: Wholesale Down Balancing Price (if total deviation positive)
        :param spot_ws: Wholesale Spot Price
        :return: For each party and timestep, their penalties
        """
        penalties = [[0 for _ in range(self.nodes)] for _ in range(self.num_time)]

        for t in range(self.num_time):
            # 1 day
            total_imb = delta_total[t]
            for i in self.C.parties:  # skip feeder automatically
                local_imb = deltas_i[t][i]
                if trivia.same_sign(total_imb, local_imb):
                    # Operate at relevant up/down reg
                    # Note that if total_imb = 0 that is handled by the one below!
                    # If your imb > 0, you get paid down regulation - tariff
                    # If your imb < 0, you need to pay for up regulation + tariff
                    pen = local_imb * (dw_reg[t] - self.C.tariff) if local_imb > 0 else local_imb * (
                            up_reg[t] + self.C.tariff)
                    penalties[t][i] = pen
                else:
                    # Operate at spot price
                    # If your imb > 0, you are selling extra power at spot price - tariff (so -ve penalty)
                    # If your imb < 0, you are buying extra power at spot price + tariff (so +ve penalty)
                    pen = local_imb * (spot_ws[t] - self.C.tariff) if local_imb > 0 else local_imb * (
                            spot_ws[t] + self.C.tariff)
                    penalties[t][i] = pen

        return penalties

    def get_errors(self, t):
        """
        Generates an error scenario!
        """
        # USE ERROR_FLOW.PY
        sd = 0.01
        non_root_node = random.sample(self.C.parties, 1)[0]
        dem_scaler = self.C.forecast["dP"][t][non_root_node]
        errors = error_flow.generate_errrors(self.C.children, self.nodes, dem_scaler, sd)

        return errors

    def get_known(self, known_nodes: list, t: int):
        """
        Takes values, known as per the Double Market and the Triggered Response
        And then Applies measurement errors to the known values as per a feasible error scenario
        Cap within the generation limits for P
        """
        errors = self.get_errors(t)

        for i in known_nodes:
            # For now, everyone behaves and does what they were scheduled to do
            self.real_values[t][i]["P"] += errors[i]["P"]
            # self.real_values[i]["Q"] += errors[i]["Q"]

            self.real_values[t][i]["p"] += errors[i]["p"]
            # self.real_values[i]["q"] += self.results_df["GenQ0[" + str(i) + "]"][t] if i in self.C.gens else 0.0

        return self.real_values[t]

    def export(self, path: str):
        """
        Store the results Dataframe as a CSV file at a path of your choosing.
        """
        export_df = self.results_df
        export_df.to_csv(path)

    def recover_geo(self, known_nodes: list, real_values: list, power: bool = True):
        """
        Set up an LP similar to above but without an Objective Problem and try to recover the unknown values
        We hold that u_root = 1 no matter, or we cannot solve the problem!

        :param known_nodes: The nodes that have submitted values (which are correct)
        :param real_values: Dicts for each known node with their P, Q, p, q, u values
        :param power: Whether we split by equal power or equal burden (do this later)
        :return:
        """
        unknown_nodes = [i for i in range(self.nodes) if i not in known_nodes]

        unknown_values = {}

        islands, dist = trivia.get_islands(self.C.outflows, self.C.feeder, known_nodes, unknown_nodes, self.nodes)

        print(islands)
        if power:
            # Equal Power
            for isle in islands:
                # Determine the honest nodes responsible for inflow and outflow
                Ancestor_isle = set([self.C.ancestor[j] for j in isle if self.C.ancestor[j] not in isle])

                Children_isle = []
                for jj in isle:
                    cc = self.C.children[jj]
                    for ii in cc:
                        if ii not in isle:
                            Children_isle.append(ii)
                Children_isle = set(Children_isle)

                # Determine total inflow and outflow
                P_sum_in = 0.0
                Q_sum_in = 0.0

                for hA in Ancestor_isle:
                    hA_hC = [uu for uu in self.C.children[hA] if uu in known_nodes]
                    hA_out_P = real_values[hA]["P"] + real_values[hA]["p"] - sum(
                        real_values[hA_cc]["P"] for hA_cc in hA_hC)
                    hA_out_Q = real_values[hA]["Q"] + real_values[hA]["q"] - sum(
                        real_values[hA_cc]["Q"] for hA_cc in hA_hC)
                    P_sum_in += hA_out_P
                    Q_sum_in += hA_out_Q

                P_sum_out = 0.0
                Q_sum_out = 0.0

                for hC in Children_isle:
                    hc_in_P = real_values[hC]["P"]
                    hc_in_Q = real_values[hC]["Q"]
                    P_sum_out += hc_in_P
                    Q_sum_out += hc_in_Q

                # Determine net injection and split it amongst them
                p_net = P_sum_out - P_sum_in
                q_net = Q_sum_out - Q_sum_in
                b = len(isle)
                p_share = p_net / b
                q_share = q_net / b

                for i in isle:
                    unknown_values[i] = {"P": -1, "p": p_share,
                                         "Q": -1, "q": q_share,
                                         "u": 0 - 1}

        return unknown_values

    def liars(self, injections, lie_occs, prices, scaler):
        """
        Implements Liars Pentalty
        For each timestep t the penalty is:
        abs(injection[t][i]["p"])*scaler[t]*lie_occs[i]*prices[t][i]

        """
        penalties = [[0.0 for _ in range(self.nodes)] for _ in range(self.num_time)]

        for t in range(self.num_time):
            for i in range(self.C.nodes):
                pen = abs(injections[t][i]["p"]) * scaler[t] * prices[i][t] * lie_occs[t][i]
                penalties[t][i] = pen

        return penalties

    def set_df(self, res_df):
        """
        Set an imported DF as our results DF
        """

        self.results_df = res_df
