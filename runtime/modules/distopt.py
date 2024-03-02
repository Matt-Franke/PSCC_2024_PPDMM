"""
Here ADMM and Secure ADMM are implemented
"""
from modules import scenario, trivia
import cvxpy as cp
import numpy
from scipy.linalg import sqrtm
import time
from multiprocessing import Process, Pipe
from math import sqrt
import json
import scipy
import subprocess
import pandas
import sys


# noinspection PyAttributeOutsideInit
class Auction:
    """
        The Optimization Problem for the Day Auction that is fully deterministic, i.e. no uncertainty
    """

    def __init__(self, C: scenario.Community, SECURE: bool, MODE: int, DEBUG: bool):
        """
        Import and derive the datasets required for the Problem
        """
        self.num_time = C.num_time
        self.nodes = C.nodes
        self.lines = C.lines
        self.C = C
        self.CC = C.CC
        self.SECURE = SECURE
        self.MODE = MODE
        self.DEBUG = DEBUG
        self.setup()

    def setup(self):
        """
        Remember, a lot of this is already in the loaded Community
        """
        self.PLOT_IT = True
        # Threshold Setup
        if self.SECURE:
            if self.MODE == 0:
                self.MAX_ITER = 500
                self.rho_pen = 15000.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.0015
                self.epsilon_flow = 3.0 / 100.0
                self.epsilon_surplus = 2.0 / 100.0
            elif self.MODE == 1:
                self.MAX_ITER = 750
                self.rho_pen = 15000.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.001
                self.epsilon_flow = 4.0 / 100.0
                self.epsilon_surplus = 2.0 / 100.0
            elif self.MODE == 2:
                self.MAX_ITER = 750
                self.rho_pen = 250.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.001
                self.epsilon_flow = 5.00 / 100.0
                self.epsilon_surplus = 2.0 / 100.0
            elif self.MODE == 3:
                self.MAX_ITER = 1000
                self.rho_pen = 100.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.05  # willing to take the hit here
                self.epsilon_flow = 7.5 / 100.0
                self.epsilon_surplus = 5.0 / 100.0
            elif self.MODE == 4:
                self.MAX_ITER = 1000
                self.rho_pen = 100.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.003  # willing to take the hit here
                self.epsilon_flow = 7.5 / 100.0
                self.epsilon_surplus = 5.0 / 100.0
            else:
                raise NotImplementedError
        else:
            if self.MODE == 0:
                self.MAX_ITER = 500
                self.rho_pen = 15000.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.001
                self.epsilon_flow = 3.0
                self.epsilon_surplus = 2.0
            elif self.MODE == 1:
                self.MAX_ITER = 500
                self.rho_pen = 15000.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.001
                self.epsilon_flow = 4.0
                self.epsilon_surplus = 2.0
            elif self.MODE == 2:
                self.MAX_ITER = 750
                self.rho_pen = 250.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.001
                self.epsilon_flow = 10.00
                self.epsilon_surplus = 2.0
            elif self.MODE == 3:
                self.MAX_ITER = 1000
                self.rho_pen = 100.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.05  # willing to take the hit here
                self.epsilon_flow = 15.0
                self.epsilon_surplus = 3.0
            elif self.MODE == 4:
                self.MAX_ITER = 1000
                self.rho_pen = 100.0
                self.epsilon_primal = 0.001
                self.epsilon_dual = 0.003  # willing to take the hit here
                self.epsilon_flow = 10.0
                self.epsilon_surplus = 5.0
            else:
                raise NotImplementedError

    def run_simulation(self, BC):
        begint = time.time()

        if self.SECURE:
            schedules, demand, lambdas, alphas, pies, balances = self.sec_admm()
        else:
            schedules, demand, lambdas, alphas, pies, balances = self.admm()

        endt = time.time()
        print(f"Took {(endt - begint) / 60} minutes to Solve Problem")
        return schedules, demand, lambdas, alphas, pies, balances

    def sec_admm(self):
        """
        SECURE ADMM GLOBAL
        """
        begint = time.time()
        MODE = self.MODE
        num_time = self.num_time
        nodes = self.nodes
        if MODE in [1, 2, 3]:
            e_g = 0.05
            z_g_f = scipy.stats.norm.ppf(1 - e_g)

            e_v = 0.01
            # Since we use 2*z_v through, and it will get mad at us otherwise
            z_v_f = 2.0 * scipy.stats.norm.ppf(1 - e_v)

            e_f = 0.01
            z_f_f = scipy.stats.norm.ppf(1 - e_f)
        elif MODE == 4:
            e_g = 0.05
            z_g_f = sqrt((1 - e_g) / e_g)
        else:
            z_g_f = 0.00
            z_v_f = 0.00
            z_f_f = 0.00

        # S calculation
        s_list = [0.0 for _ in range(num_time)]
        s2_list = numpy.zeros((num_time, nodes - 1))  # To be used as s2_list[t, :]

        if MODE:
            for t in range(num_time):
                """
                Calculate the squared sum of standard devations
                """
                var_all = [i ** 2 for i in self.C.forecast["SD"][t]]
                sum_var = sum(var_all)
                s2_list[t] = numpy.array(var_all[1:])  # all but feeder
                s_list[t] = sqrt(sum_var)

        """
        Loaders
        CC: bool, days, s_list, s2_list, r x lim gen: bool,
                     l_cost q_cost z_g Q_min_f Q_max_f Q_mid_f Pmax_f
                     Pmin_f FT, GT, i, pipe
        """
        gen = [True if i in self.C.gens else False for i in range(self.C.nodes)]

        A_list_dlmp = [[0 for _ in range(self.lines)] for _ in range(self.lines)]

        for ppp in self.C.parties:
            node_2_root = ppp
            while node_2_root != self.C.feeder:
                li = self.C.connect[node_2_root][
                    self.C.ancestor[node_2_root]]  # the line between the next node and its ancestor
                A_list_dlmp[li][ppp - 1] = 1
                node_2_root = self.C.ancestor[node_2_root]  # go one up

        FT_f = self.C.forecast["FT"]
        GT_f = self.C.forecast["GT"]

        """
        Multiplier and Globals setup
        """

        # Global Variables
        global_P = numpy.zeros((nodes, num_time))
        global_Q = numpy.zeros((nodes, num_time))
        global_U = numpy.zeros((nodes, num_time))

        if MODE in [1, 2, 3]:
            global_A = numpy.zeros((nodes, num_time))

        if MODE == 3:
            global_RF = numpy.zeros((nodes, num_time))
            global_RV = numpy.zeros((nodes, num_time))

        if MODE == 4:
            global_A = numpy.zeros((nodes, nodes, num_time))

        # THESE ARE GLOBAL CONSTANTS!!!!
        up_set = [[jjj, self.C.ancestor[jjj]] if jjj else [jjj] for jjj in range(nodes)]
        dw_set = [[jjj] + self.C.children[jjj] for jjj in range(nodes)]

        """
        1 over the number nodes that have your flow in their local variables
        for flow, the people that have your flow in their set are you and your ancestor!
        for voltage, it is you and your children
        So hilariously, it is "up_set" for voltage and the "volt^_set" for flow

        This makes sense if you refer to Mieth2019 and Dvorkin2020.
        Let U be the upstream set, which is you and your ancestor
        Let D be the downstream set, which is you and your children

        For flows: you send your flow to your D and receive flows from your U
        For voltage: you send your volt mag to your U and receive volts from your D

        """
        f_inv = [1.0 / len(up_set[k]) for k in range(nodes)]  # used by you and your ancestor: U
        v_inv = [1.0 / len(dw_set[k]) for k in range(nodes)]  # used by you and your children: D
        n_inv = 1.0 / nodes

        # NOTE: We do not actually track local values, we average straight away!!
        # Here we initialise arrays used to capture the final outputs (to be verified with global variables)
        result_l_P = numpy.zeros(num_time)
        result_l_Q = numpy.zeros(num_time)
        result_s_P = numpy.zeros(num_time)
        result_s_Q = numpy.zeros(num_time)

        gQ_result = numpy.zeros(num_time)
        B_result = numpy.zeros((nodes, num_time))  # Battery SOC (0 if not a generator or the feeder)
        gP_result = numpy.zeros((nodes, num_time))  # should be == to global_P[Children[i]]-global_P[i]
        sP_result = numpy.zeros(
            (nodes, num_time))  # check global_P[0, t]. If +/-, then sp's should be within 1% of GT/FT

        if MODE in [1, 2, 3]:
            alpha_result = numpy.zeros((nodes, num_time))  # should be == global_A
            pi_result = numpy.zeros(num_time)  # everyone gets same dual of "Alpha Balance at time t", so do consensus!
            # For passive, just average the send-in values for pi to smooth out small errors.
            # Later do consensus on bigger table!
        elif MODE == 4:
            alpha_result = numpy.zeros((nodes, nodes, num_time))  # should be == global_A
            pi_result = numpy.zeros(
                (nodes, num_time))  # everyone gets same dual of "Alpha Balance at time t", so do consensus!
            # For passive, just average the send-in values for pi to smooth out small errors.
            # Later do consensus on bigger table!

        # To make the looping easier
        nothing = [0.0 for _ in range(num_time)]
        der_node = []
        for i in range(nodes):
            if i in self.C.pv:
                DER = abs(self.C.forecast["PV"])
            elif i == self.C.wind[0]:
                DER = abs(self.C.forecast["Wind1"])
            elif i == self.C.wind[1]:
                DER = abs(self.C.forecast["Wind2"])
            else:
                DER = [0.0 for _ in range(num_time)]  # by default no renewable
            der_node.append(DER)

        dem_node = [self.C.load if i != self.C.feeder else [0.0 for _ in range(num_time)] for i in range(nodes)]

        der_array = numpy.array(der_node)
        dP_array = numpy.array(dem_node)
        total_demand = numpy.sum(dP_array, axis=0)

        print("Setup Data Bundling and Community")
        """
        Compute Network Bounds
        """
        # All battery discharge at limit + DERs with no load
        max_P_outflow = [sum([der_node[i][t] + self.C.g_P_max[i] for i in range(nodes)]) for t in range(num_time)]

        # No reactive generation in the network!!
        max_Q_outflow = [0.0 for _ in range(num_time)]

        # All batteries charge at limit + Full P Demand + No DER
        max_P_inflow = [sum([-1.0 * self.C.g_P_min[i] + dem_node[i][t] for i in range(nodes)]) for t in range(num_time)]

        # Full Q Demand
        max_Q_inflow = [sum([1.0 * self.C.pf * dem_node[i][t] for i in range(nodes)]) for t in range(num_time)]

        u_root = 1.0

        maxes = {"P_I": max_P_inflow, "P_O": max_P_outflow, "Q_I": max_Q_inflow, "Q_O": max_Q_outflow, "U0": u_root}

        final_it = self.MAX_ITER

        """
        Metrics collector
        We collect the Euclidean norm over residuals of a category for each node over all timesteps for each iteration
        """
        norm_P = numpy.zeros((nodes, self.MAX_ITER + 1))
        norm_Q = numpy.zeros((nodes, self.MAX_ITER + 1))
        norm_U = numpy.zeros((nodes, self.MAX_ITER + 1))

        if MODE:
            norm_A = numpy.zeros((nodes, self.MAX_ITER + 1))

        if MODE == 3:
            norm_RF = numpy.zeros((nodes, self.MAX_ITER + 1))
            norm_RV = numpy.zeros((nodes, self.MAX_ITER + 1))

        norm_O = numpy.zeros((1, self.MAX_ITER + 1))

        # Track the residuals we are making decisions with
        norm_PRIME = numpy.zeros((1, self.MAX_ITER + 1))
        norm_DUAL = numpy.zeros((1, self.MAX_ITER + 1))
        norm_FLOW = numpy.zeros((1, self.MAX_ITER + 1))
        norm_SURPLUS = numpy.zeros((1, self.MAX_ITER + 1))

        # Track value of globals when the problem had the best OF
        obj_best = float("inf")  # since we are going to call obj_curr < obj_best
        global_P_best = numpy.zeros((nodes, num_time))
        global_Q_best = numpy.zeros((nodes, num_time))
        global_U_best = numpy.zeros((nodes, num_time))

        if MODE in [1, 2, 3]:
            global_A_best = numpy.zeros((nodes, num_time))

        if MODE == 3:
            global_RF_best = numpy.zeros((nodes, num_time))
            global_RV_best = numpy.zeros((nodes, num_time))

        if MODE == 4:
            global_A_best = numpy.zeros((nodes, nodes, num_time))

        # Globally calculated prices]
        da_price = numpy.zeros((nodes, num_time))

        if MODE in [1, 2, 3]:
            flex_price = numpy.zeros((1, num_time))

        if MODE == 4:
            flex_price = numpy.zeros((nodes, num_time))

        gP_curr = numpy.zeros((nodes, num_time))

        norm_DA = numpy.zeros((nodes, self.MAX_ITER + 1))
        norm_FL = numpy.zeros((1, self.MAX_ITER + 1))

        # Store the Lambdas!
        lambda_P = numpy.zeros((nodes, num_time))
        if MODE in [1, 2, 3]:
            lambda_A = numpy.zeros((nodes, num_time))

        if MODE == 4:
            lambda_A = numpy.zeros((nodes, nodes, num_time))

        # Timing
        time_LOCAL = numpy.zeros((1, self.MAX_ITER + 1))
        time_GLOBAL = numpy.zeros((1, self.MAX_ITER + 1))
        """
        CREATE/ OVERWRITE PARAMETERS FILE HERE
        """
        da_store = [[0.0] * self.num_time] * self.nodes

        if self.MODE in [1, 2, 3]:
            flex_store = [0.0] * self.num_time
        elif self.MODE == 4:
            flex_store = [[0.0] * self.num_time] * self.nodes
        else:
            flex_store = [0.0] * self.num_time
        OP = 0
        dictionary = {"dw_set": dw_set, "up_set": up_set, "MODE": self.MODE, "self.DEBUG": self.DEBUG, "OP": OP, "da": da_store,
                      "flex": flex_store}
        json_object = json.dumps(dictionary)
        with open("parameters.json", "w") as outfile:
            outfile.write(json_object)

        random_port = 11443
        SSL = False
        res_smpc = "res.json"
        if SSL:
            command = [sys.executable, ".\\joint_smpc.py", "-M" + str(self.nodes), "-ssl", "-B", str(random_port),
                       "--no-log"]
        else:
            command = [sys.executable, ".\\joint_smpc.py", "-M" + str(self.nodes), "--no-log"]

        print("SMPC Setup Complete")

        """
        Spin up Nodes IN ORDER (important, since we thus know via the index which node is behind which pipe)
        """
        pipes = []
        procs = []
        print("Spawn Nodes")
        for i in range(self.nodes):
            local, remote = Pipe()
            pipes += [local]

            args_node = (self.MAX_ITER, up_set[i], dw_set[i], self.nodes, A_list_dlmp, self.C.r, self.C.pf, dem_node[i],
                         der_node[i], self.num_time, s_list, s2_list, self.C.r[i], self.C.x[i], self.C.S_max[i],
                         gen[i], self.rho_pen, self.C.lin_cost[i], self.C.quad_cost[i], z_v_f,
                         z_f_f, z_g_f, self.C.Q_min[i] * self.C.Q[i], self.C.Q_max[i] * self.C.Q[i],
                         self.C.Q_midnight[i] * self.C.Q[i], self.C.g_P_max[i],
                         self.C.g_P_min[i], FT_f, GT_f, maxes, i, self.MODE, f_inv, v_inv, n_inv, self.epsilon_primal,
                         self.epsilon_dual, self.epsilon_flow, self.epsilon_surplus, remote)
            procs += [Process(target=self.run_node_sec, args=args_node)]
            procs[-1].start()

        """
        Check constraints were established properly
        """

        dcp_cks = [pipe.recv() for pipe in pipes]  # 1 if successful, something else, otherwise
        s_dcp_cks = sum(dcp_cks)
        print("DCP CHECK: {}, Vals: {}".format(s_dcp_cks == self.nodes, dcp_cks))

        if s_dcp_cks != self.nodes:
            raise ValueError

        """
        ******************************
        GLOBAL LOOP BEGINS
        ******************************
        """
        start_LOCAL = time.time()  # start of first local iteration here
        print("Begin Nodes")
        for iteration in range(self.MAX_ITER + 1):
            # Send current iteration
            for pipe in pipes:
                pipe.send(iteration)

            # Collect status from local optimisers
            # This also means they have finished storing their state!
            stats = [pipe.recv() for pipe in pipes]  # 1 if successful, something else, otherwise

            end_LOCAL = time.time()  # end of local iteration
            time_LOCAL[0, iteration] = end_LOCAL - start_LOCAL
            start_GLOBAL = time.time()

            for n_stat in range(len(stats)):
                if stats[n_stat] == "optimal":
                    stats[n_stat] = "O"
                if stats[n_stat] == "optimal_inaccurate":
                    stats[n_stat] = "OA"

            check = sum([1 if sst == "O" or sst == "OA" else 0 for sst in stats])

            if check != self.nodes:
                # If all were optimal, then check will be sum(1 for _ in range(self.nodes))=self.nodes
                print("Someone was not optimal!")
                print(stats)
                [p2.terminate() for p2 in procs]
                raise ValueError

            """
            CALL SMPC NODE HERE and RECEIVE ITS OUTPUT
            """
            if not iteration:
                print("Begin first SMPC")

            tic = time.time()
            cmd = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            _, err = cmd.communicate()
            if cmd.returncode != 0:
                print(err)
                raise ValueError
            toc = time.time()
            print("SMPC took {} seconds or {} minutes".format(toc - tic, (toc - tic) / 60))

            """
            RETRIEVE THE RESULTS!
            """
            # Read the json file:
            with open(res_smpc, 'r') as openfile:
                # Reading from json file
                result_dict = json.load(openfile)

            # All floats
            totalOF = result_dict["zC"]
            # All arrays
            zP = numpy.array(result_dict["zP"])
            zQ = numpy.array(result_dict["zQ"])
            zU = numpy.array(result_dict["zU"])
            da_price = numpy.array(result_dict["zD"])
            da_store = result_dict["zD"][:]
            # All floats
            p_dec = result_dict["p_tot"]
            n_dec = result_dict["n_tot"]
            dec = 1 if p_dec < 0.0 and n_dec < 0.0 else 0
            r_sum = result_dict["r_tot"]
            s_sum = result_dict["s_tot"]
            total_residual = result_dict["t_tot"]

            if self.MODE:
                flex_store = result_dict["zF"][:]
                flex_price = numpy.array(result_dict["zF"])
                zA = numpy.array(result_dict["zA"])

            if self.MODE == 3:
                zRV = numpy.array(result_dict["zRV"])
                zRF = numpy.array(result_dict["zRF"])

            if iteration > 0:
                # to deal with insertion error!
                norm_O[0, iteration] = totalOF

            # Receive values from all self.nodes
            # Step 1: cast our lists of lists back to arrays!
            global_P = zP[:]
            global_Q = zQ[:]
            global_U = zU[:]

            if self.MODE:
                global_A = zA[:]

            if self.MODE == 3:
                global_RF = zRF[:]
                global_RV = zRV[:]

            # Check if the new OF value is better
            if totalOF < obj_best:
                obj_best = totalOF  # Update
                global_P_best = numpy.copy(global_P)
                global_Q_best = numpy.copy(global_Q)
                global_U_best = numpy.copy(global_U)
                if self.MODE:
                    global_A_best = numpy.copy(global_A)
                if self.MODE == 3:
                    global_RF_best = numpy.copy(global_RF)
                    global_RV_best = numpy.copy(global_RV)

            print("It. {}: OF Values: {}, stats: {}".format(iteration, totalOF, stats))

            if self.MODE:
                norm_FL[0, iteration] = numpy.linalg.norm(flex_price)

            # For now, because I only calculated one price lol
            # Update global prices using global variables
            for n in range(self.nodes):
                norm_DA[n, iteration] = numpy.linalg.norm(da_price[n])

            # SEND GLOBALS
            for n in range(self.nodes):
                use_pipe = pipes[n]  # use this pipe to talk to them!

                # We can just slice the arrays to get what we want
                # For flows, you need yours and your children, for volt, you need yours and your ancestor
                global_P_send = numpy.array([global_P[lP] for lP in dw_set[n]])
                global_Q_send = numpy.array([global_Q[lQ] for lQ in dw_set[n]])
                global_U_send = numpy.array([global_U[lU] for lU in up_set[n]])

                # Compute then send per NODE
                use_pipe.send(global_P_send)
                use_pipe.send(global_Q_send)
                use_pipe.send(global_U_send)

                if self.MODE:
                    use_pipe.send(global_A)

                if self.MODE == 3:
                    use_pipe.send(global_RF)
                    use_pipe.send(global_RV)

            primals = [pipe.recv() for pipe in pipes]
            duals = [pipe.recv() for pipe in pipes]

            """
            MAKE DECISION
            """
            if r_sum == self.nodes and s_sum == self.nodes and dec:
                terminator = 1.0
            elif iteration == self.MAX_ITER:
                terminator = 1.0
            else:
                terminator = 0.0

            rel_surplus = total_residual * 100 / sum(total_demand)

            if iteration > 1:
                norm_PRIME[0, iteration] = r_sum
                norm_DUAL[0, iteration] = s_sum
                norm_SURPLUS[0, iteration] = rel_surplus
            fact_pos = total_residual - sum(total_demand) * self.epsilon_surplus
            fact_neg = - sum(total_demand) * self.epsilon_surplus - total_residual
            statement_stopping = "CENTRAL It{}, Primal: {}, P_Sum: {}, Dual:{}, D_Sum: {}, Surplus: {}, DEC:{}, Terminate:{}"
            print(
                statement_stopping.format(iteration, max(primals), r_sum, max(duals), s_sum, rel_surplus, dec,
                                          terminator))

            """
            DOUBLE CHECK BY JUST RUNNING DECISION BLOCK AGAIN
            """
            if terminator == 1.0 and iteration != self.MAX_ITER:
                OP = 1
                dictionary = {"dw_set": dw_set, "up_set": up_set, "MODE": self.MODE, "self.DEBUG": self.DEBUG, "OP": OP,
                              "da": da_store,
                              "flex": flex_store}
                json_object = json.dumps(dictionary)
                with open("parameters.json", "w") as outfile:
                    outfile.write(json_object)

                """
                CALL SMPC FOR CONVERGENCE DECISION!
                """

                tic = time.time()
                cmd = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                _, err = cmd.communicate()
                if cmd.returncode != 0:
                    print(err)
                    raise ValueError
                toc = time.time()
                print("SMPC Decision took {} seconds or {} minutes".format(toc - tic, (toc - tic) / 60))

                # Read the json file:
                with open(res_smpc, 'r') as openfile:
                    # Reading from json file
                    result_dict = json.load(openfile)
                p_dec = result_dict["p_tot"]
                n_dec = result_dict["n_tot"]
                dec = 1 if p_dec < 0.0 and n_dec < 0.0 else 0
                r_sum = result_dict["r_tot"]
                s_sum = result_dict["s_tot"]
                total_residual = result_dict["t_tot"]
                f_dec = result_dict["f_tot"]

                if r_sum == self.nodes and s_sum == self.nodes and dec and f_dec:
                    terminator = 1.0
                else:
                    terminator = 0.0

                rel_surplus = total_residual * 100 / sum(total_demand)

                if iteration > 1:
                    norm_PRIME[0, iteration] = r_sum
                    norm_DUAL[0, iteration] = s_sum
                    norm_SURPLUS[0, iteration] = rel_surplus
                fact_pos = total_residual - sum(total_demand) * self.epsilon_surplus
                fact_neg = - sum(total_demand) * self.epsilon_surplus - total_residual
                statement_stopping = "CENTRAL It{}, Primal: {}, P_Sum: {}, Dual:{}, D_Sum: {}, F_Dec: {}, Surplus: {}, DEC:{}, Terminate:{}"
                print(
                    statement_stopping.format(iteration, max(primals), r_sum, max(duals), s_sum, f_dec, rel_surplus,
                                              dec,
                                              terminator))

                OP = 0
                dictionary = {"dw_set": dw_set, "up_set": up_set, "MODE": self.MODE, "self.DEBUG": self.DEBUG, "OP": OP,
                              "da": da_store,
                              "flex": flex_store}
                json_object = json.dumps(dictionary)
                with open("parameters.json", "w") as outfile:
                    outfile.write(json_object)

            end_GLOBAL = time.time()
            time_GLOBAL[0, iteration] = end_GLOBAL - start_GLOBAL
            start_LOCAL = time.time()  # start of local iteration here

            # SEND THE TERMINATOR!!!
            for pipe in pipes:
                pipe.send(terminator)

            if terminator == 1.0:
                """
                CALL SMPC BALANCE HERE to GET GLOBAL PRICES AND BALANCES
                """
                OP = 2
                dictionary = {"dw_set": dw_set, "up_set": up_set, "MODE": self.MODE, "self.DEBUG": self.DEBUG, "OP": OP,
                              "da": da_store,
                              "flex": flex_store}
                json_object = json.dumps(dictionary)
                with open("parameters.json", "w") as outfile:
                    outfile.write(json_object)

                lt = time.time()
                timing = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(lt))
                print("At time: " + timing)
                print("Begin SMPC BALANCE")
                tic = time.time()
                cmd = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                _, err = cmd.communicate()
                if cmd.returncode != 0:
                    print(err)
                    raise ValueError
                toc = time.time()
                print("SMPC Balance took {} seconds or {} minutes".format(toc - tic, (toc - tic) / 60))

                """
                RETRIEVE THE RESULTS!
                """
                # Read the json file:
                with open(res_smpc, 'r') as openfile:
                    # Reading from json file
                    result_dict = json.load(openfile)
                da_price_alt = numpy.array(result_dict["zD"])
                bals_SMPC = numpy.array(result_dict["zB"])
                zP_alt = numpy.array(result_dict["zP"])

                if self.MODE:
                    zA_alt = numpy.array(result_dict["zA"])
                    flex_price_alt = numpy.array(result_dict["zF"])

                # check if zP_alt and zP are the same. Same for zA and zA_alt
                if (zP == zP_alt).all():
                    if self.MODE:
                        if (zA == zA_alt).all():
                            print("Global Variables same between both SMPCs!")
                    else:
                        print("Global Variables same between both SMPCs!")

                if (da_price == da_price_alt).all():
                    if self.MODE:
                        if (flex_price == flex_price_alt).all():
                            print("Global Prices same between both SMPCs!")
                    else:
                        print("Global Prices same between both SMPCs!")

                # this part is not actually needed, but is just there for verification of the results!
                # this will be later removed with just the breaking out of the loop preserved
                final_it = iteration
                # Receive local values!!
                for n in range(self.nodes):
                    use_pipe = pipes[n]  # use this pipe to talk to them!

                    if n == 0:
                        result_l_P = use_pipe.recv()
                        result_l_Q = use_pipe.recv()
                        result_s_P = use_pipe.recv()
                        result_s_Q = use_pipe.recv()
                        gP_result[n] = result_l_P - result_s_P
                    else:
                        gP_result[n] = use_pipe.recv()

                    if gen[n]:
                        B_result[n] = use_pipe.recv()

                    sP_result[n] = use_pipe.recv()

                    lambda_P[n] = use_pipe.recv()

                    if self.MODE in [1, 2, 3]:
                        pi_result += use_pipe.recv()
                        lambda_A[n] = use_pipe.recv()

                    if self.MODE == 4:
                        pi_result += use_pipe.recv()
                        in_A = use_pipe.recv()
                        lambda_A[:, n, :] = in_A

                break

            """
            STORE DA AND FLEX PRICES
            """
            OP = 0
            dictionary = {"dw_set": dw_set, "up_set": up_set, "MODE": self.MODE, "self.DEBUG": self.DEBUG, "OP": OP,
                          "da": da_store,
                          "flex": flex_store}
            json_object = json.dumps(dictionary)
            with open("parameters.json", "w") as outfile:
                outfile.write(json_object)

            # Collect and receive the norms of the multipliers for tracking!
            # Store at iteration+1, since all start at 0 on iteration 0!
            for n in range(self.nodes):
                use_pipe = pipes[n]  # use this pipe to talk to them!

                norm_P[n, iteration] = use_pipe.recv()
                norm_Q[n, iteration] = use_pipe.recv()
                norm_U[n, iteration] = use_pipe.recv()

                if self.MODE:
                    norm_A[n, iteration] = use_pipe.recv()

                if self.MODE == 3:
                    norm_RF[n][iteration] = use_pipe.recv()
                    norm_RV[n][iteration] = use_pipe.recv()

        [p2.terminate() for p2 in procs]

        # Store Computation Times: Local and Global
        its_l = numpy.array([rr for rr in range(self.MAX_ITER + 1)])
        results_dict = {"Iteration": its_l, "Local": time_LOCAL[0], "Global": time_GLOBAL[0]}
        results_df = pandas.DataFrame(results_dict)

        endings = ["SEC_DET", "SEC_GEN_LIN", "SEC_GEN_SOC", "SEC_FULL"]
        path = "../data/Processed/results_" + endings[MODE] + "_TIME"

        results_df.to_csv(path)

        """
        ******************************
        PLOT RESIDUALS!!!
        ******************************
        """
        # Plot Active Power Balance Residuals
        if self.PLOT_IT:
            trivia.plot_residuals(norm_P[:, :final_it + 1], "P", final_it)
            trivia.plot_residuals(norm_Q[:, :final_it + 1], "Q", final_it)
            trivia.plot_residuals(norm_U[:, :final_it + 1], "U", final_it)
            trivia.plot_residuals(norm_O[:, :final_it + 1], "Cost", final_it)
            trivia.plot_residuals(norm_PRIME[:, :final_it + 1], "Prime", final_it)
            trivia.plot_residuals(norm_DUAL[:, :final_it + 1], "Dual", final_it)
            trivia.plot_residuals(norm_FLOW[:, :final_it + 1], "Flow", final_it)
            trivia.plot_residuals(norm_SURPLUS[:, :final_it + 1], "Relative Surplus", final_it)
            trivia.plot_residuals(norm_DA[:, :final_it + 1], "DA", final_it)
            trivia.plot_residuals(time_LOCAL[:, :final_it + 1], "Local Time", final_it)
            trivia.plot_residuals(time_GLOBAL[:, :final_it + 1], "Global Time", final_it)

            if self.MODE:
                trivia.plot_residuals(norm_A[:, :final_it + 1], "A", final_it)
                trivia.plot_residuals(norm_FL[:, :final_it + 1], "FLEX", final_it)

            if self.MODE == 3:
                trivia.plot_residuals(norm_RF[:, :final_it + 1], "RF", final_it)
                trivia.plot_residuals(norm_RV[:, :final_it + 1], "RV", final_it)

        """"
        Verify local results!
        """
        # gP = numpy.zeros((self.nodes, self.num_time))
        # should be == to global_P[Children[i]]-global_P[i]
        # sP = numpy.zeros((self.nodes, self.num_time))
        # check global_P[0, t]. If +/-, then sp's should be within 1% of GT/FT
        # alpha_result = numpy.zeros((self.nodes, self.num_time))
        # should be == global_A
        # pi_result = numpy.zeros((self.nodes, self.num_time))
        # everyone gets same dual of "Alpha Balance at time t", so do consensus!
        # For passive, just average the send-in values for pi to smooth out small errors.
        # Later do consensus on bigger table!

        """
        Store Results and Compute Balances for the Day
        """
        """
        Store Results and Update Settlements
        """
        lambda_0 = sP_result[0]
        shadow_prices = sP_result
        result_dP = [[] for _ in range(self.nodes)]
        result_SOC = B_result

        samePrice = [abs(lambda_0[t] - numpy.max(abs(shadow_prices[:, t]))) for t in range(self.num_time)]
        result_t_P = [sum(gP_result[1:, t]) for t in range(self.num_time)]
        result_DER = [sum(der_array[:, t]) for t in range(self.num_time)]
        result_d_P = [sum(dP_array[:, t]) for t in range(self.num_time)]
        result_d_Q = [self.C.pf * sum(dP_array[:, t]) for t in range(self.num_time)]

        # Generated Energy: Flexible Generation, DER
        # Demanded Energy: Demand, Net Battery Charge
        # All Batteries start at self.C.Q_midnight[i] * self.C.Q[i]
        start_charge = sum([self.C.Q_midnight[i] * self.C.Q[i] for i in range(self.nodes)])

        net_battery_charge = [
            sum(result_SOC[:, t] - result_SOC[:, t - 1]) if t else sum(result_SOC[:, t]) - start_charge
            for t in range(self.num_time)]

        print("Net Battery Charge: {}".format(net_battery_charge))

        flow_surplus = [sum(gP_result[:, t]) + sum(der_array[:, t]) - result_d_P[t] for t in
                        range(self.num_time)]
        total_flow_mag = [sum(abs(gP_result[:, t])) + sum(der_array[:, t]) + result_d_P[t] for t in
                          range(self.num_time)]

        rel_surplus = [flow_surplus[t] * 100 / total_flow_mag[t] for t in range(self.num_time)]
        """
        Missing results
        #floP,floQ, Battery SOC!!!
        """

        if self.MODE in [1, 2, 3]:
            # both of these are arrays that go [i][t]
            result_alpha = global_A[:]
            alpha_price = pi_result[:]
            result_st_dev = self.C.forecast["SD"]

        if self.MODE == 4:
            # both of these are arrays that go [i][t]
            result_alpha = global_A[:]
            alpha_price = pi_result[:]
            result_st_dev = self.C.forecast["SD"]

        balances = [[] for _ in range(self.nodes)]  # We know they all start with 0!
        surplus = []

        for t in range(self.num_time):
            """
            SETTLEMENT
            """
            total = 0

            if self.MODE:
                # Flex Sharing Idea Two: Actors causing uncertainty pay for it
                # Since this is day ahead, split according to day ahead forecast errors
                SD_total = sum(result_st_dev[t])

            curr_bal = 0.0
            for i in range(self.nodes):
                if t:
                    curr_bal = balances[i][t - 1]

                if i == self.C.feeder:  # Feeder
                    bal_up = shadow_prices[i] * gP_result[i, t]
                    if self.MODE in [1, 2, 3]:
                        bal_up += alpha_price[t] * result_alpha[i, t]
                    if self.MODE == 4:
                        bal_up += sum([alpha_price[ui, t] * result_alpha[ui, i, t] for ui in range(self.nodes)])

                else:  # Everyone Else
                    bal_up = shadow_prices[i][t] * (gP_result[i, t] - dem_node[i][t] + der_node[i][t])
                    if self.MODE in [1, 2, 3]:
                        bal_up += alpha_price[t] * result_alpha[i, t]
                        # Your share of total uncertainty times alpha price (which is also total payment)
                        bal_up -= alpha_price[t] * (result_st_dev[t][i] / SD_total)
                    if self.MODE == 4:
                        bal_up += sum([alpha_price[ui, t] * result_alpha[ui, i, t] for ui in range(self.nodes)])
                        bal_up -= alpha_price[i, t]  # pay for your uncertainty fully

                total += bal_up
                balances[i].append(curr_bal + bal_up)

            surplus.append(total)

        """"
        Verify local results!
        """
        # gP = numpy.zeros((self.nodes, self.num_time))
        # should be == to global_P[Children[i]]-global_P[i]
        # sP = numpy.zeros((self.nodes, self.num_time))
        # check global_P[0, t]. If +/-, then sp's should be within 1% of GT/FT
        # alpha_result = numpy.zeros((self.nodes, self.num_time))
        # should be == global_A
        # pi_result = numpy.zeros((self.nodes, self.num_time))
        # everyone gets same dual of "Alpha Balance at time t", so do consensus!
        # For passive, just average the send-in values for pi to smooth out small errors.
        # Later do consensus on bigger table!

        """
        OUTPUT AND STORAGE
        """
        final_balance = [balances[i][-1] for i in range(self.nodes)]
        print("Final Balance")
        print(final_balance)
        final_surplus = sum(final_balance)
        print("Final Surplus")
        print(final_surplus)

        print("Final Balance SMPC")
        print(bals_SMPC)
        final_surplus_SMPC = sum(bals_SMPC)
        print("Final Surplus SMPC")
        print(final_surplus_SMPC)
        print(surplus)
        print("Flow Surplus")
        print(flow_surplus)
        print("Total Flow Surplus")
        print(sum(flow_surplus))
        print("Energy Residual")
        print(total_residual)
        print("Total Relative Surplus")
        print(total_residual * 100 / sum(total_demand))
        fh_ts = pandas.date_range('11/1/2013', periods=self.num_time, freq='1H')  # generate the timestamps

        # noinspection PyDictCreation
        results_dict = {"Timestep": fh_ts, "P Feed In": result_l_P, "P Feed Out": result_s_P,
                        "P DER": result_DER,
                        "P Local Gen": result_t_P, "P Local Dem": result_d_P,
                        "Q Feed In": result_l_Q, "Q Feed Out": result_s_Q, "Q Local Dem": result_d_Q,
                        "FT": FT_f, "GT": GT_f,
                        "Surplus": surplus, "Spot": self.C.forecast["WM"]}

        # Final Surplus
        rel_l = numpy.array([total_residual for _ in range(self.num_time)])
        results_dict["Rel Surplus"] = rel_l

        if self.MODE in [1, 2, 3]:
            results_dict["PI"] = alpha_price.tolist()

        if self.MODE == 4:
            for ui in range(self.nodes):
                results_dict["PI" + str(ui)] = alpha_price[ui, :].tolist()

        for i in range(self.nodes):
            results_dict["Shadow Price " + str(i)] = shadow_prices[i].tolist()

        for i in self.C.gens:
            results_dict["DER" + str(i)] = der_node[i]
            results_dict["gP" + str(i)] = gP_result[i].tolist()
            results_dict["B" + str(i)] = result_SOC[i]

        # SOC
        if self.MODE in [1, 2, 3]:
            for i in range(self.nodes):
                results_dict["Alpha" + str(i)] = result_alpha[i].tolist()

        if self.MODE == 4:
            for ui in range(self.nodes):
                for i in range(self.nodes):
                    results_dict["Alpha" + str(i) + " for " + str(ui)] = result_alpha[ui, i, :].tolist()

        for i in range(self.nodes):
            results_dict["Balance N" + str(i)] = balances[i]

        # Add in additional trackers!
        # BEST OF and associated globals!
        final_it_l = numpy.array([final_it for _ in range(self.num_time)])
        results_dict["Num ITS"] = final_it_l
        best_of = numpy.array([obj_best for _ in range(self.num_time)])
        results_dict["BEST OF"] = best_of
        rhO_l = numpy.array([self.rho_pen for _ in range(self.num_time)])
        results_dict["Pen Factor"] = rhO_l

        for i in range(self.nodes):
            results_dict["BEST P" + str(i)] = global_P_best[i].tolist()
            results_dict["BEST Q" + str(i)] = global_Q_best[i].tolist()
            results_dict["BEST U" + str(i)] = global_U_best[i].tolist()

            if self.MODE in [1, 2, 3]:
                results_dict["BEST A" + str(i)] = global_A_best[i].tolist()

            if self.MODE == 3:
                results_dict["BEST RF" + str(i)] = global_RF_best[i].tolist()
                results_dict["BEST RV" + str(i)] = global_RV_best[i].tolist()

            if self.MODE == 4:
                for ui in range(self.nodes):
                    results_dict["BEST A" + str(i) + " for " + str(ui)] = global_A_best[ui, i, :].tolist()

        # Store Final Values for Global P and A
        for i in range(self.nodes):
            results_dict["Global P" + str(i)] = global_P[i].tolist()

            if self.MODE in [1, 2, 3]:
                results_dict["Global A" + str(i)] = global_A[i].tolist()

            if self.MODE == 4:
                for ui in range(self.nodes):
                    results_dict["Global A" + str(i) + " for " + str(ui)] = global_A[ui, i, :].tolist()

        # Store Final Values for Lambda P and A
        for i in range(self.nodes):
            results_dict["Lambda P" + str(i)] = lambda_P[i].tolist()

            if self.MODE in [1, 2, 3]:
                results_dict["Lambda A" + str(i)] = lambda_A[i].tolist()

            if self.MODE == 4:
                for ui in range(self.nodes):
                    results_dict["Lambda A" + str(i) + " for " + str(ui)] = lambda_A[ui, i, :].tolist()

        # Store Globally calculated prices
        for i in range(self.nodes):
            results_dict["Global DA" + str(i)] = da_price[i].tolist()

        if self.MODE in [1, 2, 3]:
            results_dict["Global Flex"] = flex_price[0].tolist()

        if self.MODE == 4:
            for i in range(self.nodes):
                results_dict["Global Flex" + str(i)] = flex_price[i].tolist()

        results_df = pandas.DataFrame(results_dict)
        results_df.set_index("Timestep")
        path = "../data/Processed/results_" + endings[MODE] + "_DATA"
        results_df.to_csv(path)
        endt = time.time()

        CC = True if self.MODE else False
        NODAL = True if self.MODE == 4 else False

        if self.PLOT_IT:
            trivia.plot_results(path, CC, self.num_time, NODAL=NODAL)

        print("Final it: {}, Rho:{}, Final Value: {}, Total Flow Surplus: {}".format(final_it, self.rho_pen,
                                                                                     norm_O[0, final_it],
                                                                                     sum(flow_surplus)))
        print("{} its, {:.2f} minutes, {:.4f} %, {:.6} abs.".format(final_it, (endt - begint) / 60,
                                                                    total_residual * 100 / sum(total_demand),
                                                                    sum(flow_surplus)))

        schedules = gP_result.tolist()
        demand = dem_node[:]
        lambdas = shadow_prices.tolist()
        alphas = result_alpha.tolist()
        pies = alpha_price.tolist()

        return schedules, demand, lambdas, alphas, pies, balances

    def run_node_sec(self, max_its: int, up_set_i: list, dw_set_i: list, nodes: int, A_list: list, all_r: list,
                     pf: float,
                     dem_node: list, der_node: list, nt: int, s_list: list, s2_list: list, r: float, x: float,
                     lim: float, gen_i: bool, rho_pen: float, l_cost: float, q_cost: float, z_v_f: float, z_f_f: float,
                     z_g_f: float, Q_min_f: float, Q_max_f: float, Q_mid_f: float, Pmax_f: float, Pmin_f: float,
                     FT: list,
                     GT: list, maxes, current: int, MODE: int, f_inv: list, v_inv: list, n_inv: float, e_primal: float,
                     e_dual: float, e_flow: float, e_surplus: float, pipe):
        """
        Run local optimisation for node i

        days: number of days we are doing
        init_m: what initialisation setting (Default is 0)
        s_list,s2_list: value of s, s^2 over the timesteps
        r,x,lim: resistance, reactance, and apparent power limit of incoming line
        gen_i: are we a generator or not
        lC, qC: linear and quadratic cost
        Q_min: float, Q_max: float, Q_mid: float: are the battery term in real terms, not percentage!!!
        der_node, dem_node: forecasted DER power and demand for this node
        FT,GT: DA price forecasts of wholesale market
        current: which node we are talking to
        pipe: how we communicate with central controller
        """
        num_time = nt
        path_state_node = ".\\states\\" + str(current) + ".json"
        path_it_node = ".\\it\\" + str(current) + ".json"

        # Figuring out coupling sets is done globally!
        # Setup up multipliers
        # Here we have reduced sets, we are in order of our flowset, so X[0] is us, X[1] is second entry in our flowset
        lambda_P = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="lambda_P")
        lambda_Q = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="lambda_Q")
        lambda_U = cp.Parameter((len(up_set_i), num_time), value=numpy.zeros((len(up_set_i), num_time)),
                                name="lambda_U")

        # Here we just have all nodes, so ours is at [current]
        if MODE in [1, 2, 3]:
            lambda_A = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="lambda_A")

        if MODE == 3:
            lambda_RF = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="lambda_RF")
            lambda_RV = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="lambda_RV")

        if MODE == 4:
            lambda_A = cp.Parameter((nodes, nodes, num_time), value=numpy.zeros((nodes, nodes, num_time)),
                                    name="lambda_A")

        # Set up global variables
        # Here we have reduced sets, we are in order of our flowset, so X[0] is us, X[1] is second entry in our flowset
        global_P = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="global_P")
        global_Q = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="global_Q")
        global_U = cp.Parameter((len(up_set_i), num_time), value=numpy.zeros((len(up_set_i), num_time)),
                                name="global_U")

        # Here we just have all nodes, so ours is at [current]
        if MODE in [1, 2, 3]:
            global_A = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="global_A")

        if MODE == 3:
            global_RF = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="global_RF")
            global_RV = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="global_RV")

        # Nodal flex, symmetric error
        if MODE == 4:
            # For each uncertain node U, a sheet of part factors for each node at each time step
            # so N * N * T
            global_A = cp.Parameter((nodes, nodes, num_time), value=numpy.zeros((nodes, nodes, num_time)),
                                    name="global_A")

        # PUBLIC CONSTANTS
        a1 = [1, 1, 0.2679, -0.2679, -1, -1, -1, -1, -0.2679, 0.2679, 1, 1]
        a2 = [0.2679, 1, 1, 1, 1, 0.2679, -0.2679, -1, -1, -1, -1, -0.2679]
        a3 = [-1, -1.366, -1, -1, -1.366, -1, -1, -1.366, -1, -1, -1.366, -1]
        circle_approx = len(a1)  # how many basically

        # Local Variables
        # if gen_i, the following are just ground to 0
        gP = cp.Variable(num_time, name="GenP")
        gQ = cp.Variable(num_time, name="GenQ")

        if MODE in [2, 3, 4]:
            quad_genP = cp.Variable(name="quad_genP")  # We bundle across time and do not include other nodes!

        if current:
            B = cp.Variable(num_time, name="Bat_SOC")  # Battery State of Charge
        else:
            s_P = cp.Variable(num_time, name="s_P")
            s_Q = cp.Variable(num_time, name="s_Q")
            l_P = cp.Variable(num_time, name="l_P")
            l_Q = cp.Variable(num_time, name="l_Q")

        # Coupled Variables: 1 per t for us and 1 for each of the others in the set
        # this is accessed as X[node, time]. Note that OUR variable is always first.
        fP = cp.Variable((len(dw_set_i), num_time), name="FP")  # sum(fP[1:, t]), ours at [0]
        fQ = cp.Variable((len(dw_set_i), num_time), name="FQ")  # sum(fP[1:, t]), ours at [0]
        us = cp.Variable((len(up_set_i), num_time), name="U")  # us[t, 1], ours at [0]

        # These parameters are the value of the associated variable but rounded to 8 decimals spots for SMPC usage
        fP_round = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="fP_round")
        fQ_round = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="fQ_round")
        us_round = cp.Parameter((len(up_set_i), num_time), value=numpy.zeros((len(up_set_i), num_time)),
                                name="us_round")

        # old GLOBAL values!!!!!
        global_P_old = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                    name="global_P_old")
        global_Q_old = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                    name="global_Q_old")
        global_U_old = cp.Parameter((len(up_set_i), num_time), value=numpy.zeros((len(up_set_i), num_time)),
                                    name="global_U_old")

        if MODE in [1, 2, 3]:
            # If no not gen_i nor feeder, the following will be just ground to 0.0!
            # coupled
            alpha = cp.Variable((nodes, num_time), name="Alpha")  # sum(alpha[t]), ours is at [current]
            global_A_old = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="global_A_old")
            alpha_round = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="alpha_round")

            if MODE == 3:
                # we wire the entires at [t][0] to 0!
                rho_v = cp.Variable((nodes, num_time), name="rho_v")  # all but feeder, ours at [current]
                rho_f = cp.Variable((nodes, num_time), name="rho_f")  # all but feeder, ours at [current]
                # For Penalty Function
                global_RF_old = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)),
                                             name="global_RF_old")
                global_RV_old = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)),
                                             name="global_RV_old")

                # not coupled and not used by feeder
                tv = cp.Variable((num_time,), name="t_V")
                tf = cp.Variable((num_time,), name="t_F")

        if MODE == 4:
            alpha = cp.Variable((nodes, nodes, num_time), name="Alpha")  # sum(alpha[t]), ours is at [current]
            S = cp.Variable((num_time,), name="S")
            global_A_old = cp.Parameter((nodes, nodes, num_time), value=numpy.zeros((nodes, nodes, num_time)),
                                        name="global_A_old")

        # Parameters
        resit = cp.Parameter(value=2.0 * r, name="R")
        react = cp.Parameter(value=2.0 * x, name="X")

        fLim = cp.Parameter(value=lim, name="f_max")
        u_min = cp.Parameter(value=0.95 ** 2, name="u_min")
        u_max = cp.Parameter(value=1.05 ** 2, name="u_max")
        Q_min = cp.Parameter(value=Q_min_f, name="Q_min")
        Q_max = cp.Parameter(value=Q_max_f, name="Q_max")
        Q_mid = cp.Parameter(value=Q_mid_f, name="Q_mid")
        Pmax = cp.Parameter(value=Pmax_f, name="Pmax")
        Pmin = cp.Parameter(value=Pmin_f, name="Pmin")
        uroot = cp.Parameter(value=maxes["U0"], name="u_root")
        minZero = cp.Parameter(value=0.0, name="minZero")
        maxOne = cp.Parameter(value=1.0, name="maxOne")
        rho_2 = cp.Parameter(value=abs(rho_pen / 2.0), name="rho_pen", nonneg=True)

        if MODE in [2, 3, 4]:
            # Matrix Constraint that need to be setup for both DET and CC
            # over nodes
            C_quad = numpy.diag(numpy.array([q_cost for _ in range(num_time)]))
            C_quad_sqrt = sqrtm(A=C_quad)
            # over nodes
            C_alpha = numpy.diag(numpy.array([q_cost for _ in range(nodes)]))
            C_alpha_sqrt = sqrtm(A=C_alpha)
            e = numpy.ones((nodes - 1,))
            eT = e.transpose()

        if MODE in [1, 2, 3]:
            if MODE == 3:
                # SOC Matricies
                A_dlmp = numpy.array(A_list)
                A_inv_dlmp = numpy.linalg.inv(A_dlmp)
                A_dt_dlmp = A_dlmp.transpose()

                Rd_dlmp = numpy.diag(numpy.array(all_r[1:]))  # only resistances of actual lines
                R_dlmp = A_dt_dlmp * Rd_dlmp * A_dlmp
                R_inv_dlmp = numpy.linalg.inv(R_dlmp)
                z_v = cp.Parameter(value=z_v_f, name="Z V", nonneg=True)
                z_f = cp.Parameter(value=z_f_f, name="Z F", nonneg=True)

            if MODE == 2 or MODE == 3:
                S_inv_dlmp_list = []
                for t in range(num_time):
                    # One per timestep!
                    S_dlmp = numpy.diag(s2_list[t, :])  # Sigma
                    S_inv_dlmp = sqrtm(S_dlmp)  # Sigma ^(1/2)
                    S_inv_dlmp_list.append(S_inv_dlmp)
            z_g_s_l = [z_g_f * sss for sss in s_list]
            z_g_s = cp.Parameter((num_time,), value=z_g_s_l, name="Z G s", nonneg=True)

        if MODE == 4:
            S_inv_dlmp_list = []
            for t in range(num_time):
                # One per timestep!
                S_dlmp = numpy.diag(s2_list[t, :])  # Sigma
                S_inv_dlmp = sqrtm(S_dlmp)  # Sigma ^(1/2)
                S_inv_dlmp_list.append(S_inv_dlmp)
            z = cp.Parameter(value=z_g_f, name="z Flex", nonneg=True)

        # Make Constraints
        Cons = trivia.ConstraintFactory(current)

        if current == 0:
            # Feeder has nothing to CC!
            for t in range(num_time):
                # C3: Active Power Balance
                fP_bal = fP[0, t] + gP[t] - dem_node[t] + der_node[t] == sum(fP[1:, t])
                Cons.create(fP_bal, "P Balance at Node " + str(current) + " at time " + str(t))

                # C4: Reactive Power Balance
                fQ_bal = fQ[0, t] + gQ[t] - pf * dem_node[t] == sum(fQ[1:, t])
                Cons.create(fQ_bal, "Q Balance at Node " + str(current) + " at time " + str(t))

                # feeder has no inflow and no ancestor
                Cons.create(fP[0, t] == minZero, "Voltage at Feeder at time " + str(t))
                Cons.create(fQ[0, t] == minZero, "Voltage at Feeder at time " + str(t))
                Cons.create(us[0, t] == uroot, "Voltage at Feeder at time " + str(t))

                # gP and gQ are aux for the flows
                Cons.create(gP[t] == l_P[t] - s_P[t], "P Gen at Feeder at time " + str(t))
                Cons.create(gQ[t] == l_Q[t] - s_Q[t], "Q Gen at Feeder at time " + str(t))

                # C19: Feeder Flow Bounds
                Cons.create(minZero <= s_P[t], "min SP" + " at time " + str(t))
                Cons.create(minZero <= s_Q[t], "min SQ" + " at time " + str(t))
                Cons.create(minZero <= l_P[t], "min LP" + " at time " + str(t))
                Cons.create(minZero <= l_Q[t], "min LQ" + " at time " + str(t))

                # C8&9: Active Power Limits
                Cons.create(s_P[t] <= 1.0 * abs(maxes["P_O"][t]), "P Gen Min Node " + " at time " + str(t))
                Cons.create(l_P[t] <= 1.0 * abs(maxes["P_I"][t]), "P Gen Max Node " + " at time " + str(t))

                # C10&11: Reactive Power Limits
                Cons.create(s_Q[t] <= 1.0 * abs(maxes["Q_O"][t]), "Q Gen Min Node " + " at time " + str(t))
                Cons.create(l_Q[t] <= 1.0 * abs(maxes["Q_I"][t]), "Q Gen Max Node " + " at time " + str(t))
        else:
            # Generic Constraints for all non-Feeder nodes
            for t in range(num_time):
                # No reactive generation in nodes
                Cons.create(gQ[t] == minZero, "Q Gen at Node at time " + str(t))

                # Root Voltage
                # Since the tap is a publicly known setting (and also helps convergence of the glob. prob.)
                if 0 in up_set_i and len(up_set_i) > 1:
                    Cons.create(us[1, t] == uroot, "Voltage at Feeder at time " + str(t))

                # C3: Active Power Balance
                fP_bal = fP[0, t] + gP[t] - dem_node[t] + der_node[t] == sum(fP[1:, t])
                Cons.create(fP_bal, "P Balance at Node " + str(current) + " at time " + str(t))

                # C4: Reactive Power Balance
                fQ_bal = fQ[0, t] + gQ[t] - pf * dem_node[t] == sum(fQ[1:, t])
                Cons.create(fQ_bal, "Q Balance at Node " + str(current) + " at time " + str(t))
                # C5: Voltage Balance
                u_drop = (us[0, t] == us[1, t] - resit * fP[0, t] - react * fQ[0, t])
                Cons.create(u_drop, "Voltage at Node " + str(current) + " at time " + str(t))

                if MODE != 3:
                    # C12: Flow limits (with gP, just set to 0 if there is none)
                    for la in range(circle_approx):
                        lin_approx = a1[la] * fP[0, t] + a2[la] * fQ[0, t] + a3[la] * fLim
                        Cons.create(lin_approx <= 0, "LA " + str(la) + " at time " + str(t))

                    # C6/7: Voltage Limits
                    Cons.create(us[0, t] >= u_min, "Min U at time " + str(t))
                    Cons.create(us[0, t] <= u_max, "Max U at time " + str(t))

            # GENERATION CC
            if MODE in [1, 2, 3]:
                for t in range(num_time):
                    # Generation Limits
                    if gen_i:
                        # If we are CC'ing, the active power and its flexible response need to be accounted for
                        # C8&9: Active Power Limits
                        Cons.create(gP[t] - z_g_s[t] * alpha[current, t] >= Pmin,
                                    "P Gen Min Node " + " at time " + str(t))
                        Cons.create(gP[t] + z_g_s[t] * alpha[current, t] <= Pmax,
                                    "P Gen Max Node " + " at time " + str(t))

                        if t == 0:
                            # C14: SOC at begin and end
                            Cons.create(B[t] == Q_mid - gP[t], "SOC @ Node " + " at time " + str(t))
                        else:
                            # C13: SOC capture
                            Cons.create(B[t] == B[t - 1] - gP[t], "SOC @ Node " + " at time " + str(t))
                        if t == (num_time - 1):
                            # 15: SOC at begin and end
                            Cons.create(B[t] - z_g_s[t] * alpha[current, t] >= Q_mid, "End SOC @ Node")

                        # C16,17: SOC limits
                        Cons.create(B[t] - z_g_s[t] * alpha[current, t] >= Q_min, "SOC Min at time " + str(t))
                        Cons.create(B[t] + z_g_s[t] * alpha[current, t] <= Q_max, "SOC Max at time " + str(t))
                    else:
                        # Otherwise, no battery, no flex generation
                        # Otherwise, no battery, no flexible generation
                        # C8: Set generation to 0!
                        Cons.create(gP[t] == minZero, "P Gen Min Node " + " at time " + str(t))
                        # C13: Set B to 0
                        Cons.create(B[t] == minZero, "P Gen Min Node " + " at time " + str(t))
            elif MODE == 4:
                for t in range(num_time):
                    # Generation Limits
                    if gen_i:
                        # If we are CC'ing, the active power and its flexible response need to be accounted for
                        # C8&9: Active Power Limits
                        Cons.create(gP[t] - z * S[t] >= Pmin, "P Gen Min Node " + " at time " + str(t))
                        Cons.create(gP[t] + z * S[t] <= Pmax, "P Gen Max Node " + " at time " + str(t))

                        if t == 0:
                            # C14: SOC at begin and end
                            Cons.create(B[t] == Q_mid - gP[t], "SOC @ Node " + " at time " + str(t))
                        else:
                            # C13: SOC capture
                            Cons.create(B[t] == B[t - 1] - gP[t], "SOC @ Node " + " at time " + str(t))
                        if t == (num_time - 1):
                            # 15: SOC at begin and end
                            Cons.create(B[t] - z * S[t] >= Q_mid, "End SOC @ Node")

                        # C16,17: SOC limits
                        Cons.create(B[t] - z * S[t] >= Q_min, "SOC Min at time " + str(t))
                        Cons.create(B[t] + z * S[t] <= Q_max, "SOC Max at time " + str(t))
                    else:
                        # Otherwise, no battery, no flex generation
                        # Otherwise, no battery, no flexible generation
                        # C8: Set generation to 0!
                        Cons.create(gP[t] == minZero, "P Gen Min Node " + " at time " + str(t))
                        # C13: Set B to 0
                        Cons.create(B[t] == minZero, "P Gen Min Node " + " at time " + str(t))
            else:
                for t in range(num_time):
                    if gen_i:
                        # If we have a battery
                        # No flex, no uncertainty, thus deterministic cons.
                        # C8&9: Active Power Limits
                        Cons.create(gP[t] >= Pmin, "P Gen Min Node " + " at time " + str(t))
                        Cons.create(gP[t] <= Pmax, "P Gen Max Node " + " at time " + str(t))

                        if t == 0:
                            # C14: SOC at begin and end
                            Cons.create(B[t] == Q_mid - gP[t], "SOC @ Node " + " at time " + str(t))
                        else:
                            # C13: SOC capture
                            Cons.create(B[t] == B[t - 1] - gP[t], "SOC @ Node " + " at time " + str(t))

                        if t == (num_time - 1):
                            # 15: SOC at begin and end
                            Cons.create(B[t] == Q_mid, "End SOC @ Node")

                        # C16,17: SOC limits
                        Cons.create(B[t] >= Q_min, "SOC Min at time " + str(t))
                        Cons.create(B[t] <= Q_max, "SOC Max at time " + str(t))
                    else:
                        # Otherwise, no battery, no flex generation
                        # Otherwise, no battery, no flexible generation
                        # C8: Set generation to 0!
                        Cons.create(gP[t] == minZero, "P Gen Min Node " + " at time " + str(t))
                        # C13: Set B to 0
                        Cons.create(B[t] == minZero, "P Gen Min Node " + " at time " + str(t))

            # Voltage and Flow CC
            if MODE == 3:
                for t in range(num_time):
                    # C6/7: Voltage Limits
                    Cons.create(us[0][t] - z_v * tv[t] >= u_min, "Min U at time " + str(t))
                    Cons.create(us[0][t] + z_v * tv[t] <= u_max, "Max U at time " + str(t))

                    # C12: Flow limits (with gP, just set to 0 if there is none)
                    for la in range(circle_approx):
                        lin_approx = a1[la] * (fP[0][t] + z_f * tf[t]) + a2[la] * fQ[0][t] + a3[la] * fLim
                        Cons.create(lin_approx <= 0, "LA+ " + str(la) + " at time " + str(t))

                        lin_approx = a1[la] * (fP[0][t] - z_f * tf[t]) + a2[la] * fQ[0][t] + a3[la] * fLim
                        Cons.create(lin_approx <= 0, "LA- " + str(la) + " at time " + str(t))

                    # SOC constraints and related constraints
                    # Just because our alpha is 0 does not mean we do not feel the effects of everyone else!
                    # C23: Sum for Voltage Aux
                    rrhou_LHS = sum(
                        [R_inv_dlmp[current - 1][j] * rho_v[j][t] for j in range(nodes - 1)])  # j from 1 to n
                    Cons.create(rrhou_LHS == alpha[current][t], "Sum U on alpha " + str(current) + " at time " + str(t))
                    # C25: Sum for Flow Aux
                    rrhof_LHS = sum([A_inv_dlmp[current - 1][j] * rho_f[j][t] for j in range(nodes - 1)])
                    Cons.create(rrhof_LHS == alpha[current][t], "Sum F on alpha " + str(current) + " at time " + str(t))

                    # C24: SOC for Voltage
                    x_l = R_dlmp[current - 1]
                    x_r = rho_v[current][t] * eT
                    x_lr = (x_l + x_r) @ S_inv_dlmp_list[t]
                    Cons.create(cp.SOC(tf[t], x_lr), "SOC for f at time " + str(t))

                    # C26: SOC for Flow
                    x_l = A_dlmp[current - 1]
                    x_r = rho_f[current][t] * eT
                    x_lr = (x_l + x_r) @ S_inv_dlmp_list[t]
                    Cons.create(cp.SOC(tf[t], x_lr), "SOC for f at time " + str(t))

                    # CX: Aux variables bigger than 0
                    Cons.create(minZero <= tf[t], "min t_f at time " + str(t))
                    Cons.create(minZero <= tv[t], "min t_v at time " + str(t))
                    Cons.create(minZero == rho_v[0][t],
                                "Wired rho_v " + str(t))  # since feeder does not have V or pF/pQ
                    Cons.create(minZero == rho_f[0][t],
                                "Wired rho_f " + str(t))  # since feeder does not have V or pF/pQ

        # Participation Factors: Same For All
        # Use the knowledge that by definition all the alphas must be between 0 and 1
        # Since everyone has to work with this huge block of alphas, this dramatically cuts down the search space
        # it also leaks nothing about anyone else, just a sanity check
        if MODE in [1, 2, 3]:
            for t in range(num_time):
                # Force all between 0 and 1
                for n_alpha in range(nodes):
                    if n_alpha == current and current and not gen_i:
                        # Nodes without battery that are not the feeder
                        Cons.create(alpha[current, t] == minZero,
                                    "Alpha Zero Node " + str(n_alpha) + " at time " + str(t))
                    else:
                        # C27 and 28: Lower and Upper Bounds for Flex Factor (ours is at current!)
                        Cons.create(alpha[n_alpha, t] >= minZero,
                                    "Alpha Min Node " + str(n_alpha) + " at time " + str(t))
                        Cons.create(alpha[n_alpha, t] <= maxOne,
                                    "Alpha Max Node " + str(n_alpha) + " at time " + str(t))

                # C18: Flex Factor Balance
                Cons.create(sum(alpha[:, t]) == maxOne, "Alpha Balance" + " at time " + str(t))

        if MODE == 4:
            for ui in range(num_time):
                for t in range(num_time):
                    # Force all between 0 and 1
                    for n_alpha in range(nodes):
                        if ui:
                            if n_alpha == current and current and not gen_i:
                                # Nodes without battery that are not the feeder
                                Cons.create(alpha[ui, current, t] == minZero,
                                            "Alpha Zero Node " + " at node " + str(ui) + str(
                                                n_alpha) + " at time " + str(
                                                t))
                            else:
                                # C27 and 28: Lower and Upper Bounds for Flex Factor (ours is at current!)
                                Cons.create(alpha[ui, n_alpha, t] >= minZero,
                                            "Alpha Min Node " + " at node " + str(ui) + str(
                                                n_alpha) + " at time " + str(t))
                                Cons.create(alpha[ui, n_alpha, t] <= maxOne,
                                            "Alpha Max Node " + " at node " + str(ui) + str(
                                                n_alpha) + " at time " + str(t))
                        else:
                            # feeder has no uncertainty, so no flex for it!
                            Cons.create(alpha[ui, n_alpha, t] == minZero,
                                        "Alpha Zero Node " + " at node " + str(ui) + str(n_alpha) + " at time " + str(
                                            t))

                    # C18: Flex Factor Balance
                    Cons.create(sum(alpha[ui, :, t]) == maxOne,
                                "Alpha Balance" + " at node " + str(ui) + " at time " + str(t))

        """
        OBJECTIVE
        Include all the multiplier*variables here!
        """
        # Linear cost is just for active power flow injections
        if current == 0:
            cost = sum([l_P[t] * GT[t] - s_P[t] * FT[t] for t in range(num_time)])
        else:
            cost = l_cost * sum(gP)  # Try out

        if MODE == 0 or MODE == 1:
            cost += q_cost * cp.sum_squares(gP)  # C_Q * \sum_t(l_P-s_P)^2 = C_Q * \sum_t(g_P)^2
            cost += q_cost * cp.sum_squares(gP)  # C_Q * \sum_tg_P^2

        if MODE in [2, 3, 4]:
            # Quadratic Cost Constraints for gP and Flex
            # Generation Quadratic Cost
            # C21: SOC constraint for Quadratic generation cost: Just out active generation!
            # Bundle across all timesteps, thus just do it once
            quad_r = gP
            # noinspection PyTypeChecker
            quad_x = C_quad_sqrt @ quad_r
            Cons.create(cp.SOC(quad_genP, quad_x), "Quad Cost P Gen" + " at node " + str(current))

            # Quadratic Terms
            cost += quad_genP

        if MODE == 4:
            # S is the sqrt of alpha times sigma
            # Can be captured as the norm of our alphas (at each uncertain node) times sqrt of Sigma
            for t in range(num_time):
                x_lr = alpha[:, current, t] @ S_inv_dlmp_list[t]
                Cons.create(cp.SOC(S[t], x_lr), "S at time " + str(t))

        """
        Penalty terms as per CVXPY: (rho/2)*sum_squares(local - global - multiplier)    
        This works as sum_{ij}X_{ij}^2 and thus we can have a matrix inside the ()
        Since ECOS is fighting me, accept we are adding (rho/2)*sqrt(sum_squares(local - global - multiplier)) instead
        Should still work but we can now express the second half via a SOC constraint
        Note however that SOC goes per columns, which in our case is per time! This is totally fine, just make aux a vector    

        If we do the other formulation, we need a lambda times residual entry!
        """
        # Active Flows Updates (should be a len(dw_set_i)*num_time matrix)
        pen_cost = rho_2 * cp.sum_squares(fP - global_P + lambda_P)
        # Reactive Flows Updates (should be a len(dw_set_i)*num_time matrix)
        pen_cost += rho_2 * cp.sum_squares(fQ - global_Q + lambda_Q)
        # Voltage Updates (inside should be a len(dw_set_i)*num_time matrix)
        pen_cost += rho_2 * cp.sum_squares(us - global_U + lambda_U)

        if MODE in [1, 2, 3]:
            # Just YOUR quadratic cost for alpha (since remember, we sum up the overall thing!)
            alpha_s_sum = sum([alpha[current, t] * s_list[t] for t in range(num_time)])  # either S or S^2 !!

            # Only if you have alphas
            if gen_i or current == 0:
                cost += q_cost * cp.sum_squares(alpha_s_sum)

            # Alpha Updates (should be a nodes*num_time matrix)
            pen_cost += rho_2 * cp.sum_squares(alpha - global_A + lambda_A)

        if MODE == 3:
            # Rho_f updates (should be a nodes*num_time matrix)
            pen_cost += rho_2 * cp.sum_squares(rho_f - global_RF + lambda_RF)
            # Rho_v updates (should be a nodes*num_time matrix)
            pen_cost += rho_2 * cp.sum_squares(rho_v - global_RV + lambda_RV)

        if MODE == 4:
            # Only if you have alphas
            if gen_i or current == 0:
                cost += q_cost * sum(S)

            # Alphas updates (should be a nodes*nodes*num_time matrix)
            pen_cost += rho_2 * cp.sum_squares(alpha - global_A + lambda_A)

        obj = cp.Minimize(cost + pen_cost)
        prob = cp.Problem(obj, Cons.getAll())

        chk = 1 if prob.is_dcp() else 0
        pipe.send(chk)

        # Primal is local - global of this iteration
        # Dual is global - old global!!!!!!
        primal_residual = cp.sum_squares(fP_round - global_P) + cp.sum_squares(fQ_round - global_Q) + cp.sum_squares(
            us_round - global_U)
        dual_residual = cp.sum_squares(global_P - global_P_old) + cp.sum_squares(
            global_Q - global_Q_old) + cp.sum_squares(
            global_U - global_U_old)

        if MODE in [1, 2, 3]:
            primal_residual += cp.sum_squares(alpha_round - global_A)
            dual_residual += cp.sum_squares(global_A - global_A_old)

        if MODE == 3:
            # SOC
            primal_residual += cp.sum_squares(rho_f - global_RF)
            dual_residual += cp.sum_squares(global_RF - global_RF_old)
            primal_residual += cp.sum_squares(rho_v - global_RV)
            dual_residual += cp.sum_squares(global_RF - global_RV_old)

        if MODE == 4:
            primal_residual += cp.sum_squares(alpha - global_A)
            dual_residual += cp.sum_squares(global_A - global_A_old)

        DER_node = numpy.array(der_node)
        dP = numpy.array(dem_node)
        zero_array = [[0.0 for _ in range(num_time)] for _ in range(nodes)]  # a nodes*num_time list of lists
        # Initalise this to all zeros so it fails first round then has last rounds numbers
        dictionary = {"r_i": 0,
                      "s_i": 0,
                      "de_f": 0,
                      "neg_de_f": 0,
                      "de_s": 0}

        json_object = json.dumps(dictionary)
        with open(path_it_node, "w") as outfile:
            outfile.write(json_object)

        while True:
            # receive iteration
            iteration = pipe.recv()

            if MODE == 0:
                prob.solve(ignore_dpp=True, solver=cp.OSQP)
            elif MODE == 1:
                prob.solve(ignore_dpp=True, solver=cp.OSQP, max_iter=50000)
            elif MODE == 2:
                if current:
                    prob.solve(ignore_dpp=True, solver=cp.ECOS, max_iters=750)
                else:
                    # Solve the LP to get duals
                    prob.solve(ignore_dpp=True, solver=cp.ECOS)
            elif MODE == 3:
                if current:
                    prob.solve(ignore_dpp=True, solver=cp.ECOS, max_iters=750)
                else:
                    # Solve the LP to get duals
                    prob.solve(ignore_dpp=True, solver=cp.ECOS)
            else:
                raise ValueError

            stat = prob.status

            # Update rounded parameters!
            fP_round.value = fP.value.round(decimals=8)
            fQ_round.value = fQ.value.round(decimals=8)
            us_round.value = us.value.round(decimals=8)

            if MODE:
                alpha_round.value = alpha.value.round(decimals=8)

            """
            STORAGE HERE
            """
            # SMPC 1: Calculate Combined OF
            c_i = cost.value

            # SMPC 2: Calculate global variables

            fP_scaled = zero_array[:]
            fQ_scaled = zero_array[:]
            us_scaled = zero_array[:]

            # Insert flows, scaled by the relevant f_inv
            for idx, nn in enumerate(dw_set_i):
                fP_scaled[nn] = list(
                    fP_round.value[idx] * f_inv[nn])  # retrieve our fP value for node nn and scaled it by its f_inv
                fQ_scaled[nn] = list(
                    fQ_round.value[idx] * f_inv[nn])  # retrieve our fQ value for node nn and scaled it by its f_inv

            # Insert voltages, scaled by the relevant v_inv
            for idx, nn in enumerate(up_set_i):
                us_scaled[nn] = list(
                    us_round.value[idx] * v_inv[nn])  # retrieve our fQ value for node nn and scaled it by its f_inv

            if MODE in [1, 2, 3]:
                alpha_s = alpha_round.value * n_inv
                alpha_ind = numpy.absolute(alpha_round.value)
                alpha_scaled = zero_array[:]

                for ii in range(nodes):
                    alpha_scaled[ii] = list(alpha_s[ii])

                alpha_unscaled = list(alpha_ind[current])

            if MODE == 3:
                # rho_v and rho_f have the same shape as alpha so do the same things!
                # The value at the feeder should be zero, but everyone is submitting values so we keep n_inv
                rho_v_s = rho_v.value * n_inv
                rho_f_s = rho_f.value * n_inv
                rho_v_scaled = zero_array[:]
                rho_f_scaled = zero_array[:]

                for ii in range(nodes):
                    rho_v_scaled[ii] = list(rho_v_s[ii])
                    rho_f_scaled[ii] = list(rho_f_s[ii])

            if MODE == 4:
                alpha_s = (alpha.value * n_inv).round(decimals=8)
                alpha_ind = numpy.absolute(alpha.value.round(decimals=8))
                alpha_scaled = [zero_array[:] for _ in range(nodes)]
                alpha_unscaled = zero_array[:]

                for ui in range(nodes):
                    for ii in range(nodes):
                        alpha_scaled[ui][ii] = list(alpha_s[ui, ii, :])

                    alpha_unscaled[ui] = list(alpha_ind[ui, current, :])

            # SMPC 3: Calculate global prices
            rho_resP = list(rho_pen * (fP.value[0, :] + gP.value + DER_node - dP -
                                       sum(fP.value[j, :] for j in range(len(dw_set_i)) if j != 0)))
            if MODE in [1, 2, 3]:
                rho_resA = (rho_pen * (numpy.sum(alpha.value, axis=0) - numpy.ones((1, num_time))))[0].tolist()

            if MODE == 4:
                rho_resA = (rho_pen * (numpy.sum(alpha.value, axis=1) - numpy.ones((nodes, num_time))))[0].tolist()

            # SMPC 5: Calculate Balances
            ni = list(gP.value - dP + DER_node)

            if MODE:
                if current:
                    our_SD = s2_list[:, current - 1]
                else:
                    our_SD = numpy.zeros((num_time,))

                total_SD = numpy.sum(s2_list, axis=1)

                sd_div = numpy.divide(our_SD, total_SD).round(decimals=8)
                shareSD = list(sd_div)

            if MODE in [1, 2, 4]:
                dictionary = {"c_i": c_i,
                              "fP_scaled": fP_scaled,
                              "fQ_scaled": fQ_scaled,
                              "us_scaled": us_scaled,
                              "alpha_scaled": alpha_scaled,
                              "rho_resP": rho_resP,
                              "rho_resA": rho_resA,
                              "num_time": num_time,
                              "ni": ni,
                              "alpha": alpha_unscaled,
                              "rho": rho_pen,
                              "shareSD": shareSD}
            elif MODE == 3:
                dictionary = {"c_i": c_i,
                              "fP_scaled": fP_scaled,
                              "fQ_scaled": fQ_scaled,
                              "us_scaled": us_scaled,
                              "alpha_scaled": alpha_scaled,
                              "rho_v_scaled": rho_v_scaled,
                              "rho_f_scaled": rho_f_scaled,
                              "rho_resP": rho_resP,
                              "rho_resA": rho_resA,
                              "num_time": num_time,
                              "ni": ni,
                              "alpha": alpha_unscaled,
                              "rho": rho_pen,
                              "shareSD": shareSD}
            else:
                dictionary = {"c_i": c_i,
                              "fP_scaled": fP_scaled,
                              "fQ_scaled": fQ_scaled,
                              "us_scaled": us_scaled,
                              "rho_resP": rho_resP,
                              "num_time": num_time,
                              "ni": ni,
                              "rho": rho_pen}

            json_object = json.dumps(dictionary)
            with open(path_state_node, "w") as outfile:
                outfile.write(json_object)

            # send in status (We use this to signal to the main script that we are ready to move to SMPC!)
            pipe.send(stat)

            """
            RECEIVE GLOBALS
            """
            # We get things back in the correct shape for us locally already!!!
            # So we can just immediately store it!
            global_P.value = pipe.recv()
            global_Q.value = pipe.recv()
            global_U.value = pipe.recv()

            if MODE:
                global_A.value = pipe.recv()

            if MODE == 3:
                global_RF.value = pipe.recv()
                global_RV.value = pipe.recv()

            # receive updated global variables and update multipliers while you wait for your next turn
            # updates as follows: local_new - global_new
            # Active Flows Updates (should be a len(up_set_i)*num_time matrix)
            lambda_P.value += fP_round.value - global_P.value
            # Reactive Flows Updates (should be a len(up_set_i)*num_time matrix)
            lambda_Q.value += fQ_round.value - global_Q.value
            # Voltage Updates (inside should be a len(dw_set_i)*num_time matrix)
            lambda_U.value += us_round.value - global_U.value

            if MODE in [1, 2, 3]:
                # Alpha is a nodes*num_time matrix
                lambda_A.value += alpha_round.value - global_A.value

            if MODE == 3:
                # Rho_f, Rho_v are all nodes*num_time matrices
                lambda_RF.value += rho_f.value - global_RF.value
                lambda_RV.value += rho_v.value - global_RV.value

            if MODE == 4:
                lambda_A.value = alpha.value - global_A.value

            r_i = 1 if primal_residual.value * sqrt(nodes) <= e_primal else 0
            s_i = 1 if dual_residual.value * sqrt(nodes) <= e_dual else 0

            # SMPC 4: Evaluate Stopping Criteria
            de_s = e_surplus * dP
            de_f = e_flow * dP
            neg_de_f = -1.0 * de_f

            dictionary = {"r_i": r_i,
                          "s_i": s_i,
                          "de_f": sum(de_f),
                          "neg_de_f": sum(neg_de_f),
                          "de_s": sum(de_s)}

            json_object = json.dumps(dictionary)
            with open(path_it_node, "w") as outfile:
                outfile.write(json_object)

            pipe.send(primal_residual.value * sqrt(nodes))
            pipe.send(dual_residual.value * sqrt(nodes))

            """
            RECEIVING  TERMINATE DECISION!
            Only if not terminated do we receive global_{P,Q,U} (and if MODE) global_A
            """
            # receive if we terminate or not!
            terminator = pipe.recv()

            if terminator == 1.0:
                # Send in net injection and nodal
                if current == 0:
                    pipe.send(l_P.value)
                    pipe.send(l_Q.value)
                    pipe.send(s_P.value)
                    pipe.send(s_Q.value)

                else:
                    pipe.send(gP.value)
                if gen_i:
                    pipe.send(B.value)

                # Send Shadow Prices
                sp_list = numpy.array(
                    [Cons.getCon("P Balance at Node " + str(current) + " at time " + str(t)).dual_value for t in
                     range(num_time)])

                sp = -1.0 * sp_list
                pipe.send(sp)
                # Send lambda P
                pipe.send(lambda_P.value[0, :])  # just ours

                if MODE in [1, 2, 3]:
                    # Send Flex Prices
                    pi = numpy.array(
                        [Cons.getCon("Alpha Balance" + " at time " + str(t)).dual_value for t in range(num_time)])
                    pipe.send(pi)
                    # Send lambda A
                    pipe.send(lambda_A.value[current, :])  # just ours

                if MODE == 4:
                    # Send Flex Prices
                    pi = numpy.array(
                        [[Cons.getCon("Alpha Balance" + " at node " + str(ui) + " at time " + str(t)).dual_value for t
                          in
                          range(num_time)] for ui in range(nodes)])
                    pipe.send(pi)
                    # Send lambda A
                    pipe.send(lambda_A.value[:, current, :])  # just ours

                # end the loop!
                break

            """
            Update norms and "old" global values.
            Send in the new norms.        
            """
            # Take the norm (which is just the Frobenius norm and thus can also handle matrices!) and send to global
            # This should all be floats!
            norm_P = numpy.linalg.norm(fP.value - global_P.value)
            norm_Q = numpy.linalg.norm(fQ.value - global_Q.value)
            norm_U = numpy.linalg.norm(us.value - global_U.value)

            if MODE:
                norm_A = numpy.linalg.norm(alpha.value - global_A.value)

            if MODE == 3:
                norm_RF = numpy.linalg.norm(rho_f.value - global_RF.value)
                norm_RV = numpy.linalg.norm(rho_v.value - global_RV.value)

            if MODE == 4:
                norm_A = numpy.linalg.norm(alpha.value - global_A.value)

            # First compute then send to not waste time!
            pipe.send(norm_P)
            pipe.send(norm_Q)
            pipe.send(norm_U)

            if MODE:
                pipe.send(norm_A)

            if MODE == 3:
                pipe.send(norm_RF)
                pipe.send(norm_RV)

            # Finally, update old global values for new ones
            # Need copy to avoid Python shenanigans
            global_P_old.value = numpy.copy(global_P.value)
            global_Q_old.value = numpy.copy(global_Q.value)
            global_U_old.value = numpy.copy(global_U.value)

            if MODE in [1, 2, 3]:
                global_A_old.value = numpy.copy(global_A.value)

            if MODE == 3:
                global_RF_old.value = numpy.copy(global_RF.value)
                global_RV_old.value = numpy.copy(global_RV.value)

            if MODE == 4:
                global_A_old.value = numpy.copy(global_A.value)

    def admm(self):
        """
        SECURE ADMM GLOBAL
        """
        begint = time.time()
        MODE = self.MODE
        num_time = self.num_time
        nodes = self.nodes
        """
            Chance Constraining!!!
        """

        if MODE in [1, 2, 3]:
            e_g = 0.05
            z_g_f = scipy.stats.norm.ppf(1 - e_g)

            e_v = 0.01
            # Since we use 2*z_v through, and it will get mad at us otherwise
            z_v_f = 2.0 * scipy.stats.norm.ppf(1 - e_v)

            e_f = 0.01
            z_f_f = scipy.stats.norm.ppf(1 - e_f)
        elif MODE == 4:
            e_g = 0.05
            z_g_f = sqrt((1 - e_g) / e_g)
            z_v_f = 0.00
            z_f_f = 0.00
        else:
            z_g_f = 0.00
            z_v_f = 0.00
            z_f_f = 0.00

        # S calculation
        s_list = [0.0 for _ in range(num_time)]
        s2_list = numpy.zeros((num_time, nodes - 1))  # To be used as s2_list[t, :]

        if MODE:
            for t in range(num_time):
                """
                Calculate the squared sum of standard devations
                """
                var_all = [i ** 2 for i in self.C.forecast["SD"][t]]
                sum_var = sum(var_all)
                s2_list[t] = numpy.array(var_all[1:])  # all but feeder
                s_list[t] = sqrt(sum_var)

        """
        Loaders
        CC: bool, days, s_list, s2_list, r x lim gen: bool,
                     l_cost q_cost z_g Q_min_f Q_max_f Q_mid_f Pmax_f
                     Pmin_f FT, GT, i, pipe
        """
        gen = [True if i in self.C.gens else False for i in range(self.C.nodes)]

        A_list_dlmp = [[0 for _ in range(self.lines)] for _ in range(self.lines)]

        for ppp in self.C.parties:
            node_2_root = ppp
            while node_2_root != self.C.feeder:
                li = self.C.connect[node_2_root][
                    self.C.ancestor[node_2_root]]  # the line between the next node and its ancestor
                A_list_dlmp[li][ppp - 1] = 1
                node_2_root = self.C.ancestor[node_2_root]  # go one up

        FT_f = self.C.forecast["FT"]
        GT_f = self.C.forecast["GT"]

        """
        Multiplier and Globals setup
        """

        # Global Variables
        global_P = numpy.zeros((nodes, num_time))
        global_Q = numpy.zeros((nodes, num_time))
        global_U = numpy.zeros((nodes, num_time))

        if MODE in [1, 2, 3]:
            global_A = numpy.zeros((nodes, num_time))

        if MODE == 3:
            global_RF = numpy.zeros((nodes, num_time))
            global_RV = numpy.zeros((nodes, num_time))

        if MODE == 4:
            global_A = numpy.zeros((nodes, nodes, num_time))

        # THESE ARE GLOBAL CONSTANTS!!!!
        up_set = [[jjj, self.C.ancestor[jjj]] if jjj else [jjj] for jjj in range(nodes)]
        dw_set = [[jjj] + self.C.children[jjj] for jjj in range(nodes)]

        """
        1 over the number nodes that have your flow in their local variables
        for flow, the people that have your flow in their set are you and your ancestor!
        for voltage, it is you and your children
        So hilariously, it is "up_set" for voltage and the "volt^_set" for flow

        This makes sense if you refer to Mieth2019 and Dvorkin2020.
        Let U be the upstream set, which is you and your ancestor
        Let D be the downstream set, which is you and your children

        For flows: you send your flow to your D and receive flows from your U
        For voltage: you send your volt mag to your U and receive volts from your D

        """
        f_inv = [1.0 / len(up_set[k]) for k in range(nodes)]  # used by you and your ancestor: U
        v_inv = [1.0 / len(dw_set[k]) for k in range(nodes)]  # used by you and your children: D
        n_inv = 1.0 / nodes

        # NOTE: We do not actually track local values, we average straight away!!
        # Here we initialise arrays used to capture the final outputs (to be verified with global variables)
        result_l_P = numpy.zeros(num_time)
        result_l_Q = numpy.zeros(num_time)
        result_s_P = numpy.zeros(num_time)
        result_s_Q = numpy.zeros(num_time)

        gQ_result = numpy.zeros(num_time)
        B_result = numpy.zeros((nodes, num_time))  # Battery SOC (0 if not a generator or the feeder)
        gP_result = numpy.zeros((nodes, num_time))  # should be == to global_P[Children[i]]-global_P[i]
        sP_result = numpy.zeros(
            (nodes, num_time))  # check global_P[0, t]. If +/-, then sp's should be within 1% of GT/FT

        if MODE in [1, 2, 3]:
            alpha_result = numpy.zeros((nodes, num_time))  # should be == global_A
            pi_result = numpy.zeros(
                num_time)  # everyone gets same dual of "Alpha Balance at time t", so do consensus!
            # For passive, just average the send-in values for pi to smooth out small errors.
            # Later do consensus on bigger table!
        elif MODE == 4:
            alpha_result = numpy.zeros((nodes, nodes, num_time))  # should be == global_A
            pi_result = numpy.zeros(
                (nodes, num_time))  # everyone gets same dual of "Alpha Balance at time t", so do consensus!
            # For passive, just average the send-in values for pi to smooth out small errors.
            # Later do consensus on bigger table!

        # To make the looping easier
        nothing = [0.0 for _ in range(num_time)]
        der_node = []
        for i in range(nodes):
            if i in self.C.pv:
                DER = abs(self.C.forecast["PV"])
            elif i == self.C.wind[0]:
                DER = abs(self.C.forecast["Wind1"])
            elif i == self.C.wind[1]:
                DER = abs(self.C.forecast["Wind2"])
            else:
                DER = [0.0 for _ in range(num_time)]  # by default no renewable
            der_node.append(DER)

        dem_node = [self.C.load if i != self.C.feeder else [0.0 for _ in range(num_time)] for i in range(nodes)]

        der_array = numpy.array(der_node)
        dP_array = numpy.array(dem_node)
        total_demand = numpy.sum(dP_array, axis=0)

        print("Setup Data Bundling and Community")
        """
        Compute Network Bounds
        """
        # All battery discharge at limit + DERs with no load
        max_P_outflow = [sum([der_node[i][t] + self.C.g_P_max[i] for i in range(nodes)]) for t in range(num_time)]

        # No reactive generation in the network!!
        max_Q_outflow = [0.0 for _ in range(num_time)]

        # All batteries charge at limit + Full P Demand + No DER
        max_P_inflow = [sum([-1.0 * self.C.g_P_min[i] + dem_node[i][t] for i in range(nodes)]) for t in range(num_time)]

        # Full Q Demand
        max_Q_inflow = [sum([1.0 * self.C.pf * dem_node[i][t] for i in range(nodes)]) for t in range(num_time)]

        u_root = 1.0

        maxes = {"P_I": max_P_inflow, "P_O": max_P_outflow, "Q_I": max_Q_inflow, "Q_O": max_Q_outflow, "U0": u_root}

        final_it = self.MAX_ITER

        """
        Metrics collector
        We collect the Euclidean norm over residuals of a category for each node over all timesteps for each iteration
        """
        norm_P = numpy.zeros((nodes, self.MAX_ITER + 1))
        norm_Q = numpy.zeros((nodes, self.MAX_ITER + 1))
        norm_U = numpy.zeros((nodes, self.MAX_ITER + 1))

        if MODE:
            norm_A = numpy.zeros((nodes, self.MAX_ITER + 1))

        if MODE == 3:
            norm_RF = numpy.zeros((nodes, self.MAX_ITER + 1))
            norm_RV = numpy.zeros((nodes, self.MAX_ITER + 1))

        norm_O = numpy.zeros((1, self.MAX_ITER + 1))

        # Track the residuals we are making decisions with
        norm_PRIME = numpy.zeros((1, self.MAX_ITER + 1))
        norm_DUAL = numpy.zeros((1, self.MAX_ITER + 1))
        norm_FLOW = numpy.zeros((1, self.MAX_ITER + 1))
        norm_SURPLUS = numpy.zeros((1, self.MAX_ITER + 1))

        # Track value of globals when the problem had the best OF
        obj_best = float("inf")  # since we are going to call obj_curr < obj_best
        global_P_best = numpy.zeros((nodes, num_time))
        global_Q_best = numpy.zeros((nodes, num_time))
        global_U_best = numpy.zeros((nodes, num_time))

        if MODE in [1, 2, 3]:
            global_A_best = numpy.zeros((nodes, num_time))

        if MODE == 3:
            global_RF_best = numpy.zeros((nodes, num_time))
            global_RV_best = numpy.zeros((nodes, num_time))

        if MODE == 4:
            global_A_best = numpy.zeros((nodes, nodes, num_time))

        # Globally calculated prices]
        da_price = numpy.zeros((nodes, num_time))

        if MODE in [1, 2, 3]:
            flex_price = numpy.zeros((1, num_time))

        if MODE == 4:
            flex_price = numpy.zeros((nodes, num_time))

        gP_curr = numpy.zeros((nodes, num_time))

        norm_DA = numpy.zeros((nodes, self.MAX_ITER + 1))
        norm_FL = numpy.zeros((1, self.MAX_ITER + 1))

        # Store the Lambdas!
        lambda_P = numpy.zeros((nodes, num_time))
        if MODE in [1, 2, 3]:
            lambda_A = numpy.zeros((nodes, num_time))

        if MODE == 4:
            lambda_A = numpy.zeros((nodes, nodes, num_time))

        # Timing
        time_LOCAL = numpy.zeros((1, self.MAX_ITER + 1))
        time_GLOBAL = numpy.zeros((1, self.MAX_ITER + 1))

        """
            Spin up Nodes IN ORDER (important, since we thus know via the index which node is behind which pipe)
        """
        pipes = []
        procs = []
        print("Spawn Nodes")
        for i in range(self.nodes):
            local, remote = Pipe()
            pipes += [local]

            args_node = (self.MAX_ITER, up_set[i], dw_set[i], self.nodes, A_list_dlmp, self.C.r, self.C.pf, dem_node[i],
                         der_node[i], self.num_time, s_list, s2_list, self.C.r[i], self.C.x[i], self.C.S_max[i],
                         gen[i], self.rho_pen, self.C.lin_cost[i], self.C.quad_cost[i], z_v_f,
                         z_f_f, z_g_f, self.C.Q_min[i] * self.C.Q[i], self.C.Q_max[i] * self.C.Q[i],
                         self.C.Q_midnight[i] * self.C.Q[i], self.C.g_P_max[i],
                         self.C.g_P_min[i], FT_f, GT_f, maxes, i, self.MODE, remote)
            procs += [Process(target=self.run_node, args=args_node)]
            procs[-1].start()

        """
        Check constraints were established properly
        """

        dcp_cks = [pipe.recv() for pipe in pipes]  # 1 if successful, something else, otherwise
        s_dcp_cks = sum(dcp_cks)
        print("DCP CHECK: {}, Vals: {}".format(s_dcp_cks == self.nodes, dcp_cks))

        if s_dcp_cks != self.nodes:
            raise ValueError

        """
        ******************************
        GLOBAL LOOP BEGINS
        ******************************
        """
        start_LOCAL = time.time()  # start of first local iteration here
        print("Begin Nodes")
        for iteration in range(self.MAX_ITER + 1):
            # Send current iteration
            for pipe in pipes:
                pipe.send(iteration)

            # Collect status from local optimisers
            stats = [pipe.recv() for pipe in pipes]  # 1 if successful, something else, otherwise

            end_LOCAL = time.time()  # end of local iteration
            time_LOCAL[0, iteration] = end_LOCAL - start_LOCAL
            start_GLOBAL = time.time()

            for n_stat in range(len(stats)):
                if stats[n_stat] == "optimal":
                    stats[n_stat] = "O"
                if stats[n_stat] == "optimal_inaccurate":
                    stats[n_stat] = "OA"

            check = sum([1 if sst == "O" or sst == "OA" else 0 for sst in stats])

            if check != self.nodes:
                # If all were optimal, then check will be sum(1 for _ in range(self.nodes))=self.nodes
                print("Someone was not optimal!")
                print(stats)
                [p2.terminate() for p2 in procs]
                raise ValueError

            valss = [pipe.recv() for pipe in pipes]  # 1 if successful, something else, otherwise
            obj_curr = sum(valss)
            if iteration > 0:
                # to deal with insertion error!
                norm_O[0, iteration] = obj_curr

            # Check if the new OF value is better
            if obj_curr < obj_best:
                obj_best = obj_curr  # Update
                global_P_best = numpy.copy(global_P)
                global_Q_best = numpy.copy(global_Q)
                global_U_best = numpy.copy(global_U)
                if self.MODE:
                    global_A_best = numpy.copy(global_A)
                if self.MODE == 3:
                    global_RF_best = numpy.copy(global_RF)
                    global_RV_best = numpy.copy(global_RV)

            # Receive values from all self.nodes
            # Step 1: revert global arrays back to 0 for the following to work!
            global_P = numpy.zeros((self.nodes, self.num_time))
            global_Q = numpy.zeros((self.nodes, self.num_time))
            global_U = numpy.zeros((self.nodes, self.num_time))
            if self.MODE in [1, 2, 3]:
                global_A = numpy.zeros((self.nodes, self.num_time))

            if self.MODE == 3:
                global_RF = numpy.zeros((self.nodes, self.num_time))
                global_RV = numpy.zeros((self.nodes, self.num_time))

            if self.MODE == 4:
                global_A = numpy.zeros((self.nodes, self.nodes, self.num_time))

            # total active power residual
            total_residual = numpy.zeros(self.num_time)

            # Step 2: We get from a node n the tuple (i, X[i]) which we divide by len(coupled set) and add into global_X[i]
            # Calculate global variables by averaging the associated local variables
            for n in range(self.nodes):
                use_pipe = pipes[n]  # use this pipe to talk to them!

                # Logic for all: Global Variables X[n] = sum_i(local_values[i])/len(set_X(n))
                # This would require us to store local values, so we decompose the above as
                # sum_i(local_values[i]/len(set_X(n)) and thus we can increment the global array and
                # Then, discard local values to save us from using a self.nodes*self.nodes*self.num_time array
                # whose values change every iter.!

                # Active Flow
                fP_n = use_pipe.recv()
                for ref_node, vals in fP_n:
                    global_P[ref_node] += f_inv[ref_node] * vals

                # Global DA Price
                gP_curr[n] = use_pipe.recv()  # receive gP
                # Do all timestep in one go via vector addition and subtraction
                local_resP = fP_n[0][1] + gP_curr[n] + der_array[n, :] - dP_array[n, :] - sum(
                    fP_n[l_f][1] for l_f in range(1, len(dw_set[n])))

                net_injection = gP_curr[n] + der_array[n, :] - dP_array[n, :]
                total_residual += net_injection  # sum of all net injections should be 0
                da_price[n] += self.rho_pen * local_resP

                # Reactive Flow
                fQ_n = use_pipe.recv()
                for ref_node, vals in fQ_n:
                    global_Q[ref_node] += f_inv[ref_node] * vals

                # Voltage Mags
                us_n = use_pipe.recv()
                for ref_node, vals in us_n:
                    global_U[ref_node] += v_inv[ref_node] * vals

                if self.MODE in [1, 2, 3]:
                    # here we need to add arrays to arrays, so we just do that and multiply ONCE at the end!
                    # Alpha
                    alpha_n = use_pipe.recv()
                    global_A += alpha_n
                    local_resA = numpy.sum(alpha_n, axis=0) - numpy.ones((1, self.num_time))
                    flex_price += self.rho_pen * local_resA

                if self.MODE == 3:
                    # Rho F
                    rho_f_n = use_pipe.recv()
                    global_RF += rho_f_n
                    # Rho V
                    rho_v_n = use_pipe.recv()
                    global_RV += rho_v_n

                if self.MODE == 4:
                    alpha_n = use_pipe.recv()
                    global_A += alpha_n
                    local_resA = numpy.sum(alpha_n, axis=1) - numpy.ones((self.nodes, self.num_time))
                    flex_price += self.rho_pen * local_resA

            if self.MODE in [1, 2, 3]:
                global_A *= n_inv
                # Do it here, since we need to do it only once
                global_resA = numpy.sum(global_A, axis=0) - numpy.ones((1, self.num_time))
                flex_price -= self.rho_pen * self.nodes * global_resA
                norm_FL[0, iteration] = numpy.linalg.norm(flex_price)

            if self.MODE == 3:
                global_RF *= n_inv
                global_RV *= n_inv

            if self.MODE == 4:
                global_A *= n_inv
                # Do it here, since we need to do it only once
                global_resA = numpy.sum(global_A, axis=1) - numpy.ones((self.nodes, self.num_time))
                flex_price -= self.rho_pen * self.nodes * global_resA
                norm_FL[0, iteration] = numpy.linalg.norm(flex_price)

            # Update global prices using global variables
            for n in range(self.nodes):
                global_resP = global_P[n, :] + gP_curr[n] + der_array[n, :] - dP_array[n, :] - sum(
                    global_P[l_f, :] for l_f in dw_set[n] if l_f != n)
                da_price[n] -= self.rho_pen * global_resP
                norm_DA[n, iteration] = numpy.linalg.norm(da_price[n])

            print("It. {}: OF Values: {}, stats: {}".format(iteration, obj_curr, stats))
            # Send back global variables
            # Compute and send to save space
            # AS ARRAYS!
            for n in range(self.nodes):
                use_pipe = pipes[n]  # use this pipe to talk to them!

                # We can just slice the arrays to get what we want
                # For flows, you need yours and your children, for volt, you need yours and your ancestor
                global_P_send = numpy.array([global_P[lP] for lP in dw_set[n]])
                global_Q_send = numpy.array([global_Q[lQ] for lQ in dw_set[n]])
                global_U_send = numpy.array([global_U[lU] for lU in up_set[n]])

                # Compute then send per NODE
                use_pipe.send(global_P_send)
                use_pipe.send(global_Q_send)
                use_pipe.send(global_U_send)

                if self.MODE:
                    use_pipe.send(global_A)

                if self.MODE == 3:
                    use_pipe.send(global_RF)
                    use_pipe.send(global_RV)

            # Collect and receive the norms of the multipliers for tracking!
            # Store at iteration+1, since all start at 0 on iteration 0!
            for n in range(self.nodes):
                use_pipe = pipes[n]  # use this pipe to talk to them!

                norm_P[n, iteration] = use_pipe.recv()
                norm_Q[n, iteration] = use_pipe.recv()
                norm_U[n, iteration] = use_pipe.recv()

                if self.MODE:
                    norm_A[n, iteration] = use_pipe.recv()
                if self.MODE == 3:
                    norm_RF[n][iteration] = use_pipe.recv()
                    norm_RV[n][iteration] = use_pipe.recv()

            # Get r_i and s_i from all
            # Check if they are all below a fixed threshold epsilon
            # If yes, terminate!
            primal_rezzies = [p.recv() for p in pipes]
            dual_rezzies = [p.recv() for p in pipes]
            rel_residual = [abs(r / d) for r, d in zip(total_residual, total_demand)]  # for tracking

            r_dec = 0
            s_dec = 0
            sqrt_N = sqrt(self.nodes)
            sqrt_T = sqrt(self.num_time)

            for iii in range(self.nodes):
                r_dec += 1 if primal_rezzies[iii] * sqrt_N <= self.epsilon_primal else 0
                s_dec += 1 if dual_rezzies[iii] * sqrt_N <= self.epsilon_dual else 0

            res_dec = 0
            for ttt in range(self.num_time):
                # Check both ways!
                dP_ep = total_demand[ttt] * self.epsilon_flow
                res_dec += 1 if dP_ep >= total_residual[ttt] * sqrt_T >= -1.0 * dP_ep else 0

            rel_surplus = sum(total_residual) * 100
            dP_surplus = sum(total_demand) * self.epsilon_surplus
            surplus_dec = 1 if dP_surplus >= rel_surplus >= -1.0 * dP_surplus else 0

            if iteration > 1:
                norm_PRIME[0, iteration] = max(primal_rezzies) * sqrt_N
                norm_DUAL[0, iteration] = max(dual_rezzies) * sqrt_N
                norm_FLOW[0, iteration] = max(rel_residual) * sqrt_T
                norm_SURPLUS[0, iteration] = rel_surplus

            if r_dec != self.nodes or s_dec != self.nodes or res_dec != self.num_time or surplus_dec != 1:
                terminator = 0.0  # need all to fulfill this, thus otherwise do not terminate
            elif iteration + 1 == self.MAX_ITER:
                terminator = 1.0  # need all to fulfill this, thus otherwise do not terminate
            else:
                terminator = 1.0

            statement_stopping = "CENTRAL It{}, Primal: {}, Dual:{}, Flow: {}, Surplus: {}, Terminate:{}"
            print(statement_stopping.format(iteration, max(primal_rezzies) * sqrt_N, max(dual_rezzies) * sqrt_N,
                                            max(rel_residual) * sqrt_T, rel_surplus, terminator))

            end_GLOBAL = time.time()
            time_GLOBAL[0, iteration] = end_GLOBAL - start_GLOBAL
            start_LOCAL = time.time()  # start of local iteration here

            for pipe in pipes:
                pipe.send(terminator)

            if terminator == 1.0:
                final_it = iteration
                # Receive local values!!
                for n in range(self.nodes):
                    use_pipe = pipes[n]  # use this pipe to talk to them!

                    if n == 0:
                        result_l_P = use_pipe.recv()
                        result_l_Q = use_pipe.recv()
                        result_s_P = use_pipe.recv()
                        result_s_Q = use_pipe.recv()
                        gP_result[n] = result_l_P - result_s_P
                    else:
                        gP_result[n] = use_pipe.recv()

                    if gen[n]:
                        B_result[n] = use_pipe.recv()

                    sP_result[n] = use_pipe.recv()

                    lambda_P[n] = use_pipe.recv()

                    if self.MODE in [1, 2, 3]:
                        pi_result += use_pipe.recv()
                        lambda_A[n] = use_pipe.recv()

                    if self.MODE == 4:
                        pi_result += use_pipe.recv()
                        in_A = use_pipe.recv()
                        lambda_A[:, n, :] = in_A

                break

        [p2.terminate() for p2 in procs]

        # Store Computation Times: Local and Global
        its_l = numpy.array([rr for rr in range(self.MAX_ITER + 1)])
        results_dict = {"Iteration": its_l, "Local": time_LOCAL[0], "Global": time_GLOBAL[0]}
        results_df = pandas.DataFrame(results_dict)

        endings = ["DET", "GEN_LIN", "GEN_SOC", "FULL"]
        path = "../data/Processed/results_" + endings[MODE] + "_TIME"

        results_df.to_csv(path)

        """
        ******************************
        PLOT RESIDUALS!!!
        ******************************
        """
        # Plot Active Power Balance Residuals
        if self.PLOT_IT:
            trivia.plot_residuals(norm_P[:, :final_it + 1], "P", final_it)
            trivia.plot_residuals(norm_Q[:, :final_it + 1], "Q", final_it)
            trivia.plot_residuals(norm_U[:, :final_it + 1], "U", final_it)
            trivia.plot_residuals(norm_O[:, :final_it + 1], "Cost", final_it)
            trivia.plot_residuals(norm_PRIME[:, :final_it + 1], "Prime", final_it)
            trivia.plot_residuals(norm_DUAL[:, :final_it + 1], "Dual", final_it)
            trivia.plot_residuals(norm_FLOW[:, :final_it + 1], "Flow", final_it)
            trivia.plot_residuals(norm_SURPLUS[:, :final_it + 1], "Relative Surplus", final_it)
            trivia.plot_residuals(norm_DA[:, :final_it + 1], "DA", final_it)
            trivia.plot_residuals(time_LOCAL[:, :final_it + 1], "Local Time", final_it)
            trivia.plot_residuals(time_GLOBAL[:, :final_it + 1], "Global Time", final_it)

            if self.MODE:
                trivia.plot_residuals(norm_A[:, :final_it + 1], "A", final_it)
                trivia.plot_residuals(norm_FL[:, :final_it + 1], "FLEX", final_it)

            if self.MODE == 3:
                trivia.plot_residuals(norm_RF[:, :final_it + 1], "RF", final_it)
                trivia.plot_residuals(norm_RV[:, :final_it + 1], "RV", final_it)

        """"
        Verify local results!
        """
        # gP = numpy.zeros((self.nodes, self.num_time))
        # should be == to global_P[Children[i]]-global_P[i]
        # sP = numpy.zeros((self.nodes, self.num_time))
        # check global_P[0, t]. If +/-, then sp's should be within 1% of GT/FT
        # alpha_result = numpy.zeros((self.nodes, self.num_time))
        # should be == global_A
        # pi_result = numpy.zeros((self.nodes, self.num_time))
        # everyone gets same dual of "Alpha Balance at time t", so do consensus!
        # For passive, just average the send-in values for pi to smooth out small errors.
        # Later do consensus on bigger table!

        """
        Store Results and Compute Balances for the Day
        """
        """
        Store Results and Update Settlements
        """
        lambda_0 = sP_result[0]
        shadow_prices = sP_result
        result_dP = [[] for _ in range(self.nodes)]
        result_SOC = B_result

        samePrice = [abs(lambda_0[t] - numpy.max(abs(shadow_prices[:, t]))) for t in range(self.num_time)]
        result_t_P = [sum(gP_result[1:, t]) for t in range(self.num_time)]
        result_DER = [sum(der_array[:, t]) for t in range(self.num_time)]
        result_d_P = [sum(dP_array[:, t]) for t in range(self.num_time)]
        result_d_Q = [self.C.pf * sum(dP_array[:, t]) for t in range(self.num_time)]

        # Generated Energy: Flexible Generation, DER
        # Demanded Energy: Demand, Net Battery Charge
        # All Batteries start at self.C.Q_midnight[i] * self.C.Q[i]
        start_charge = sum([self.C.Q_midnight[i] * self.C.Q[i] for i in range(self.nodes)])

        net_battery_charge = [
            sum(result_SOC[:, t] - result_SOC[:, t - 1]) if t else sum(result_SOC[:, t]) - start_charge
            for t in range(self.num_time)]

        print("Net Battery Charge: {}".format(net_battery_charge))

        flow_surplus = [sum(gP_result[:, t]) + sum(der_array[:, t]) - result_d_P[t] for t in
                        range(self.num_time)]
        total_flow_mag = [sum(abs(gP_result[:, t])) + sum(der_array[:, t]) + result_d_P[t] for t in
                          range(self.num_time)]

        rel_surplus = [flow_surplus[t] * 100 / total_flow_mag[t] for t in range(self.num_time)]
        """
        Missing results
        #floP,floQ, Battery SOC!!!
        """

        if self.MODE in [1, 2, 3]:
            # both of these are arrays that go [i][t]
            result_alpha = global_A[:]
            alpha_price = pi_result[:]
            result_st_dev = self.C.forecast["SD"]

        if self.MODE == 4:
            # both of these are arrays that go [i][t]
            result_alpha = global_A[:]
            alpha_price = pi_result[:]
            result_st_dev = self.C.forecast["SD"]

        balances = [[] for _ in range(self.nodes)]  # We know they all start with 0!
        surplus = []

        for t in range(self.num_time):
            """
            SETTLEMENT
            """
            total = 0

            if self.MODE:
                # Flex Sharing Idea Two: Actors causing uncertainty pay for it
                # Since this is day ahead, split according to day ahead forecast errors
                SD_total = sum(result_st_dev[t])

            curr_bal = 0.0
            for i in range(self.nodes):
                if t:
                    curr_bal = balances[i][t - 1]

                if i == self.C.feeder:  # Feeder
                    bal_up = lambda_0[t] * gP_result[i, t]
                    if self.MODE in [1, 2, 3]:
                        bal_up += alpha_price[t] * result_alpha[i, t]
                    if self.MODE == 4:
                        bal_up += sum([alpha_price[ui, t] * result_alpha[ui, i, t] for ui in range(self.nodes)])

                else:  # Everyone Else
                    bal_up = lambda_0[t] * (gP_result[i, t] - dem_node[i][t] + der_node[i][t])
                    if self.MODE in [1, 2, 3]:
                        bal_up += alpha_price[t] * result_alpha[i, t]
                        # Your share of total uncertainty times alpha price (which is also total payment)
                        bal_up -= alpha_price[t] * (result_st_dev[t][i] / SD_total)
                    if self.MODE == 4:
                        bal_up += sum([alpha_price[ui, t] * result_alpha[ui, i, t] for ui in range(self.nodes)])
                        bal_up -= alpha_price[i, t]  # pay for your uncertainty fully

                total += bal_up
                balances[i].append(curr_bal + bal_up)

            surplus.append(total)

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
        results_dict = {"Timestep": fh_ts, "P Feed In": result_l_P, "P Feed Out": result_s_P,
                        "P DER": result_DER,
                        "P Local Gen": result_t_P, "P Local Dem": result_d_P,
                        "Q Feed In": result_l_Q, "Q Feed Out": result_s_Q, "Q Local Dem": result_d_Q,
                        "FT": FT_f, "GT": GT_f,
                        "Surplus": surplus, "Spot": self.C.forecast["WM"]}

        # Final Surplus
        results_dict["Rel Surplus"] = total_residual.tolist()

        if self.MODE in [1, 2, 3]:
            results_dict["PI"] = alpha_price.tolist()

        if self.MODE == 4:
            for ui in range(self.nodes):
                results_dict["PI" + str(ui)] = alpha_price[ui, :].tolist()

        for i in range(self.nodes):
            results_dict["Shadow Price " + str(i)] = shadow_prices[i].tolist()

        for i in self.C.gens:
            results_dict["DER" + str(i)] = der_node[i]
            results_dict["gP" + str(i)] = gP_result[i].tolist()
            results_dict["B" + str(i)] = result_SOC[i]

        # SOC
        if self.MODE in [1, 2, 3]:
            for i in range(self.nodes):
                results_dict["Alpha" + str(i)] = result_alpha[i].tolist()

        if self.MODE == 4:
            for ui in range(self.nodes):
                for i in range(self.nodes):
                    results_dict["Alpha" + str(i) + " for " + str(ui)] = result_alpha[ui, i, :].tolist()

        for i in range(self.nodes):
            results_dict["Balance N" + str(i)] = balances[i]

        # Add in additional trackers!
        # BEST OF and associated globals!
        final_it_l = numpy.array([final_it for _ in range(self.num_time)])
        results_dict["Num ITS"] = final_it_l
        best_of = numpy.array([obj_best for _ in range(self.num_time)])
        results_dict["BEST OF"] = best_of
        rhO_l = numpy.array([self.rho_pen for _ in range(self.num_time)])
        results_dict["Pen Factor"] = rhO_l

        for i in range(self.nodes):
            results_dict["BEST P" + str(i)] = global_P_best[i].tolist()
            results_dict["BEST Q" + str(i)] = global_Q_best[i].tolist()
            results_dict["BEST U" + str(i)] = global_U_best[i].tolist()

            if self.MODE in [1, 2, 3]:
                results_dict["BEST A" + str(i)] = global_A_best[i].tolist()

            if self.MODE == 3:
                results_dict["BEST RF" + str(i)] = global_RF_best[i].tolist()
                results_dict["BEST RV" + str(i)] = global_RV_best[i].tolist()

            if self.MODE == 4:
                for ui in range(self.nodes):
                    results_dict["BEST A" + str(i) + " for " + str(ui)] = global_A_best[ui, i, :].tolist()

        # Store Final Values for Global P and A
        for i in range(self.nodes):
            results_dict["Global P" + str(i)] = global_P[i].tolist()

            if self.MODE in [1, 2, 3]:
                results_dict["Global A" + str(i)] = global_A[i].tolist()

            if self.MODE == 4:
                for ui in range(self.nodes):
                    results_dict["Global A" + str(i) + " for " + str(ui)] = global_A[ui, i, :].tolist()

        # Store Final Values for Lambda P and A
        for i in range(self.nodes):
            results_dict["Lambda P" + str(i)] = lambda_P[i].tolist()

            if self.MODE in [1, 2, 3]:
                results_dict["Lambda A" + str(i)] = lambda_A[i].tolist()

            if self.MODE == 4:
                for ui in range(self.nodes):
                    results_dict["Lambda A" + str(i) + " for " + str(ui)] = lambda_A[ui, i, :].tolist()

        # Store Globally calculated prices
        for i in range(self.nodes):
            results_dict["Global DA" + str(i)] = da_price[i].tolist()

        if self.MODE in [1, 2, 3]:
            results_dict["Global Flex"] = flex_price[0].tolist()

        if self.MODE == 4:
            for i in range(self.nodes):
                results_dict["Global Flex" + str(i)] = flex_price[i].tolist()

        results_df = pandas.DataFrame(results_dict)
        results_df.set_index("Timestep")
        path = "../data/Processed/results_" + endings[MODE] + "_DATA"
        results_df.to_csv(path)
        endt = time.time()
        CC = True if self.MODE else False
        NODAL = True if self.MODE == 4 else False

        if self.PLOT_IT:
            trivia.plot_results(path, CC, self.num_time, NODAL=NODAL)

        print("Final it: {}, Rho:{}, Final Value: {}, Total Flow Surplus: {}".format(final_it, self.rho_pen,
                                                                                     norm_O[0, final_it],
                                                                                     sum(flow_surplus)))
        print("{} its, {:.2f} minutes, {:.4f} %, {:.6} abs.".format(final_it, (endt - begint) / 60,
                                                                    sum(total_residual) * 100 / sum(total_demand),
                                                                    sum(flow_surplus)))
        schedules = gP_result.tolist()
        demand = dem_node[:]
        lambdas = shadow_prices.tolist()
        alphas = result_alpha.tolist()
        pies = alpha_price.tolist()

        return schedules, demand, lambdas, alphas, pies, balances

    def run_node(self, max_its: int, up_set_i: list, dw_set_i: list, nodes: int, A_list: list, all_r: list, pf: float,
                 dem_node: list, der_node: list, nt: int, s_list: list, s2_list: list, r: float, x: float,
                 lim: float, gen_i: bool, rho_pen: float, l_cost: float, q_cost: float, z_v_f: float, z_f_f: float,
                 z_g_f: float, Q_min_f: float, Q_max_f: float, Q_mid_f: float, Pmax_f: float, Pmin_f: float, FT: list,
                 GT: list, maxes, current: int, MODE: int, pipe):
        """
        Run local optimisation for node i

        days: number of days we are doing
        init_m: what initialisation setting (Default is 0)
        s_list,s2_list: value of s, s^2 over the timesteps
        r,x,lim: resistance, reactance, and apparent power limit of incoming line
        gen_i: are we a generator or not
        lC, qC: linear and quadratic cost
        Q_min: float, Q_max: float, Q_mid: float: are the battery term in real terms, not percentage!!!
        der_node, dem_node: forecasted DER power and demand for this node
        FT,GT: DA price forecasts of wholesale market
        current: which node we are talking to
        pipe: how we communicate with central controller
        """
        num_time = nt
        # Figuring out coupling sets is done globally!
        # Setup up multipliers
        # Here we have reduced sets, we are in order of our flowset, so X[0] is us, X[1] is second entry in our flowset
        lambda_P = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="lambda_P")
        lambda_Q = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="lambda_Q")
        lambda_U = cp.Parameter((len(up_set_i), num_time), value=numpy.zeros((len(up_set_i), num_time)),
                                name="lambda_U")

        # Here we just have all nodes, so ours is at [current]
        if MODE in [1, 2, 3]:
            lambda_A = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="lambda_A")

        if MODE == 3:
            lambda_RF = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="lambda_RF")
            lambda_RV = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="lambda_RV")

        if MODE == 4:
            lambda_A = cp.Parameter((nodes, nodes, num_time), value=numpy.zeros((nodes, nodes, num_time)),
                                    name="lambda_A")

        # Set up global variables
        # Here we have reduced sets, we are in order of our flowset, so X[0] is us, X[1] is second entry in our flowset
        global_P = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="global_P")
        global_Q = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                name="global_Q")
        global_U = cp.Parameter((len(up_set_i), num_time), value=numpy.zeros((len(up_set_i), num_time)),
                                name="global_U")

        # Here we just have all nodes, so ours is at [current]
        if MODE in [1, 2, 3]:
            global_A = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="global_A")

        if MODE == 3:
            global_RF = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="global_RF")
            global_RV = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="global_RV")

        # Nodal flex, symmetric error
        if MODE == 4:
            # For each uncertain node U, a sheet of part factors for each node at each time step
            # so N * N * T
            global_A = cp.Parameter((nodes, nodes, num_time), value=numpy.zeros((nodes, nodes, num_time)),
                                    name="global_A")

        # PUBLIC CONSTANTS
        a1 = [1, 1, 0.2679, -0.2679, -1, -1, -1, -1, -0.2679, 0.2679, 1, 1]
        a2 = [0.2679, 1, 1, 1, 1, 0.2679, -0.2679, -1, -1, -1, -1, -0.2679]
        a3 = [-1, -1.366, -1, -1, -1.366, -1, -1, -1.366, -1, -1, -1.366, -1]
        circle_approx = len(a1)  # how many basically

        # Local Variables
        # if gen_i, the following are just ground to 0
        gP = cp.Variable(num_time, name="GenP")
        gQ = cp.Variable(num_time, name="GenQ")

        if MODE in [2, 3, 4]:
            quad_genP = cp.Variable(name="quad_genP")  # We bundle across time and do not include other nodes!

        if current:
            B = cp.Variable(num_time, name="Bat_SOC")  # Battery State of Charge
        else:
            s_P = cp.Variable(num_time, name="s_P")
            s_Q = cp.Variable(num_time, name="s_Q")
            l_P = cp.Variable(num_time, name="l_P")
            l_Q = cp.Variable(num_time, name="l_Q")

        # Coupled Variables: 1 per t for us and 1 for each of the others in the set
        # this is accessed as X[node, time]. Note that OUR variable is always first.
        fP = cp.Variable((len(dw_set_i), num_time), name="FP")  # sum(fP[1:, t]), ours at [0]
        fQ = cp.Variable((len(dw_set_i), num_time), name="FQ")  # sum(fP[1:, t]), ours at [0]
        us = cp.Variable((len(up_set_i), num_time), name="U")  # us[t, 1], ours at [0]

        # old GLOBAL values!!!!!
        global_P_old = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                    name="global_P_old")
        global_Q_old = cp.Parameter((len(dw_set_i), num_time), value=numpy.zeros((len(dw_set_i), num_time)),
                                    name="global_Q_old")
        global_U_old = cp.Parameter((len(up_set_i), num_time), value=numpy.zeros((len(up_set_i), num_time)),
                                    name="global_U_old")

        if MODE in [1, 2, 3]:
            # If no not gen_i nor feeder, the following will be just ground to 0.0!
            # coupled
            alpha = cp.Variable((nodes, num_time), name="Alpha")  # sum(alpha[t]), ours is at [current]
            global_A_old = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)), name="global_A_old")

            if MODE == 3:
                # we wire the entires at [t][0] to 0!
                rho_v = cp.Variable((nodes, num_time), name="rho_v")  # all but feeder, ours at [current]
                rho_f = cp.Variable((nodes, num_time), name="rho_f")  # all but feeder, ours at [current]
                # For Penalty Function
                global_RF_old = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)),
                                             name="global_RF_old")
                global_RV_old = cp.Parameter((nodes, num_time), value=numpy.zeros((nodes, num_time)),
                                             name="global_RV_old")

                # not coupled and not used by feeder
                tv = cp.Variable((num_time,), name="t_V")
                tf = cp.Variable((num_time,), name="t_F")

        if MODE == 4:
            alpha = cp.Variable((nodes, nodes, num_time), name="Alpha")  # sum(alpha[t]), ours is at [current]
            S = cp.Variable((num_time,), name="S")
            global_A_old = cp.Parameter((nodes, nodes, num_time), value=numpy.zeros((nodes, nodes, num_time)),
                                        name="global_A_old")

        # Parameters
        resit = cp.Parameter(value=2.0 * r, name="R")
        react = cp.Parameter(value=2.0 * x, name="X")

        fLim = cp.Parameter(value=lim, name="f_max")
        u_min = cp.Parameter(value=0.95 ** 2, name="u_min")
        u_max = cp.Parameter(value=1.05 ** 2, name="u_max")
        Q_min = cp.Parameter(value=Q_min_f, name="Q_min")
        Q_max = cp.Parameter(value=Q_max_f, name="Q_max")
        Q_mid = cp.Parameter(value=Q_mid_f, name="Q_mid")
        Pmax = cp.Parameter(value=Pmax_f, name="Pmax")
        Pmin = cp.Parameter(value=Pmin_f, name="Pmin")
        uroot = cp.Parameter(value=maxes["U0"], name="u_root")
        minZero = cp.Parameter(value=0.0, name="minZero")
        maxOne = cp.Parameter(value=1.0, name="maxOne")
        rho_2 = cp.Parameter(value=abs(rho_pen / 2.0), name="rho_pen", nonneg=True)

        if MODE in [2, 3, 4]:
            # Matrix Constraint that need to be setup for both DET and CC
            # over nodes
            C_quad = numpy.diag(numpy.array([q_cost for _ in range(num_time)]))
            C_quad_sqrt = sqrtm(A=C_quad)
            # over nodes
            C_alpha = numpy.diag(numpy.array([q_cost for _ in range(nodes)]))
            C_alpha_sqrt = sqrtm(A=C_alpha)
            e = numpy.ones((nodes - 1,))
            eT = e.transpose()

        if MODE in [1, 2, 3]:
            if MODE == 3:
                # SOC Matricies
                A_dlmp = numpy.array(A_list)
                A_inv_dlmp = numpy.linalg.inv(A_dlmp)
                A_dt_dlmp = A_dlmp.transpose()

                Rd_dlmp = numpy.diag(numpy.array(all_r[1:]))  # only resistances of actual lines
                R_dlmp = A_dt_dlmp * Rd_dlmp * A_dlmp
                R_inv_dlmp = numpy.linalg.inv(R_dlmp)
                z_v = cp.Parameter(value=z_v_f, name="Z V", nonneg=True)
                z_f = cp.Parameter(value=z_f_f, name="Z F", nonneg=True)

            if MODE == 2 or MODE == 3:
                S_inv_dlmp_list = []
                for t in range(num_time):
                    # One per timestep!
                    S_dlmp = numpy.diag(s2_list[t, :])  # Sigma
                    S_inv_dlmp = sqrtm(S_dlmp)  # Sigma ^(1/2)
                    S_inv_dlmp_list.append(S_inv_dlmp)
            z_g_s_l = [z_g_f * sss for sss in s_list]
            z_g_s = cp.Parameter((num_time,), value=z_g_s_l, name="Z G s", nonneg=True)

        if MODE == 4:
            S_inv_dlmp_list = []
            for t in range(num_time):
                # One per timestep!
                S_dlmp = numpy.diag(s2_list[t, :])  # Sigma
                S_inv_dlmp = sqrtm(S_dlmp)  # Sigma ^(1/2)
                S_inv_dlmp_list.append(S_inv_dlmp)
            z = cp.Parameter(value=z_g_f, name="z Flex", nonneg=True)

        # Make Constraints
        Cons = trivia.ConstraintFactory(current)

        if current == 0:
            # Feeder has nothing to CC!
            for t in range(num_time):
                # C3: Active Power Balance
                fP_bal = fP[0, t] + gP[t] - dem_node[t] + der_node[t] == sum(fP[1:, t])
                Cons.create(fP_bal, "P Balance at Node " + str(current) + " at time " + str(t))

                # C4: Reactive Power Balance
                fQ_bal = fQ[0, t] + gQ[t] - pf * dem_node[t] == sum(fQ[1:, t])
                Cons.create(fQ_bal, "Q Balance at Node " + str(current) + " at time " + str(t))

                # feeder has no inflow and no ancestor
                Cons.create(fP[0, t] == minZero, "Voltage at Feeder at time " + str(t))
                Cons.create(fQ[0, t] == minZero, "Voltage at Feeder at time " + str(t))
                Cons.create(us[0, t] == uroot, "Voltage at Feeder at time " + str(t))

                # gP and gQ are aux for the flows
                Cons.create(gP[t] == l_P[t] - s_P[t], "P Gen at Feeder at time " + str(t))
                Cons.create(gQ[t] == l_Q[t] - s_Q[t], "Q Gen at Feeder at time " + str(t))

                # C19: Feeder Flow Bounds
                Cons.create(minZero <= s_P[t], "min SP" + " at time " + str(t))
                Cons.create(minZero <= s_Q[t], "min SQ" + " at time " + str(t))
                Cons.create(minZero <= l_P[t], "min LP" + " at time " + str(t))
                Cons.create(minZero <= l_Q[t], "min LQ" + " at time " + str(t))

                # C8&9: Active Power Limits
                Cons.create(s_P[t] <= 1.0 * abs(maxes["P_O"][t]), "P Gen Min Node " + " at time " + str(t))
                Cons.create(l_P[t] <= 1.0 * abs(maxes["P_I"][t]), "P Gen Max Node " + " at time " + str(t))

                # C10&11: Reactive Power Limits
                Cons.create(s_Q[t] <= 1.0 * abs(maxes["Q_O"][t]), "Q Gen Min Node " + " at time " + str(t))
                Cons.create(l_Q[t] <= 1.0 * abs(maxes["Q_I"][t]), "Q Gen Max Node " + " at time " + str(t))
        else:
            # Generic Constraints for all non-Feeder nodes
            for t in range(num_time):
                # No reactive generation in nodes
                Cons.create(gQ[t] == minZero, "Q Gen at Node at time " + str(t))

                # Root Voltage
                # Since the tap is a publicly known setting (and also helps convergence of the glob. prob.)
                if 0 in up_set_i and len(up_set_i) > 1:
                    Cons.create(us[1, t] == uroot, "Voltage at Feeder at time " + str(t))

                # C3: Active Power Balance
                fP_bal = fP[0, t] + gP[t] - dem_node[t] + der_node[t] == sum(fP[1:, t])
                Cons.create(fP_bal, "P Balance at Node " + str(current) + " at time " + str(t))

                # C4: Reactive Power Balance
                fQ_bal = fQ[0, t] + gQ[t] - pf * dem_node[t] == sum(fQ[1:, t])
                Cons.create(fQ_bal, "Q Balance at Node " + str(current) + " at time " + str(t))
                # C5: Voltage Balance
                u_drop = (us[0, t] == us[1, t] - resit * fP[0, t] - react * fQ[0, t])
                Cons.create(u_drop, "Voltage at Node " + str(current) + " at time " + str(t))

                if MODE != 3:
                    # C12: Flow limits (with gP, just set to 0 if there is none)
                    for la in range(circle_approx):
                        lin_approx = a1[la] * fP[0, t] + a2[la] * fQ[0, t] + a3[la] * fLim
                        Cons.create(lin_approx <= 0, "LA " + str(la) + " at time " + str(t))

                    # C6/7: Voltage Limits
                    Cons.create(us[0, t] >= u_min, "Min U at time " + str(t))
                    Cons.create(us[0, t] <= u_max, "Max U at time " + str(t))

            # GENERATION CC
            if MODE in [1, 2, 3]:
                for t in range(num_time):
                    # Generation Limits
                    if gen_i:
                        # If we are CC'ing, the active power and its flexible response need to be accounted for
                        # C8&9: Active Power Limits
                        Cons.create(gP[t] - z_g_s[t] * alpha[current, t] >= Pmin,
                                    "P Gen Min Node " + " at time " + str(t))
                        Cons.create(gP[t] + z_g_s[t] * alpha[current, t] <= Pmax,
                                    "P Gen Max Node " + " at time " + str(t))

                        if t == 0:
                            # C14: SOC at begin and end
                            Cons.create(B[t] == Q_mid - gP[t], "SOC @ Node " + " at time " + str(t))
                        else:
                            # C13: SOC capture
                            Cons.create(B[t] == B[t - 1] - gP[t], "SOC @ Node " + " at time " + str(t))
                        if t == (num_time - 1):
                            # 15: SOC at begin and end
                            Cons.create(B[t] - z_g_s[t] * alpha[current, t] >= Q_mid, "End SOC @ Node")

                        # C16,17: SOC limits
                        Cons.create(B[t] - z_g_s[t] * alpha[current, t] >= Q_min, "SOC Min at time " + str(t))
                        Cons.create(B[t] + z_g_s[t] * alpha[current, t] <= Q_max, "SOC Max at time " + str(t))
                    else:
                        # Otherwise, no battery, no flex generation
                        # Otherwise, no battery, no flexible generation
                        # C8: Set generation to 0!
                        Cons.create(gP[t] == minZero, "P Gen Min Node " + " at time " + str(t))
                        # C13: Set B to 0
                        Cons.create(B[t] == minZero, "P Gen Min Node " + " at time " + str(t))
            elif MODE == 4:
                for t in range(num_time):
                    # Generation Limits
                    if gen_i:
                        # If we are CC'ing, the active power and its flexible response need to be accounted for
                        # C8&9: Active Power Limits
                        Cons.create(gP[t] - z * S[t] >= Pmin, "P Gen Min Node " + " at time " + str(t))
                        Cons.create(gP[t] + z * S[t] <= Pmax, "P Gen Max Node " + " at time " + str(t))

                        if t == 0:
                            # C14: SOC at begin and end
                            Cons.create(B[t] == Q_mid - gP[t], "SOC @ Node " + " at time " + str(t))
                        else:
                            # C13: SOC capture
                            Cons.create(B[t] == B[t - 1] - gP[t], "SOC @ Node " + " at time " + str(t))
                        if t == (num_time - 1):
                            # 15: SOC at begin and end
                            Cons.create(B[t] - z * S[t] >= Q_mid, "End SOC @ Node")

                        # C16,17: SOC limits
                        Cons.create(B[t] - z * S[t] >= Q_min, "SOC Min at time " + str(t))
                        Cons.create(B[t] + z * S[t] <= Q_max, "SOC Max at time " + str(t))
                    else:
                        # Otherwise, no battery, no flex generation
                        # Otherwise, no battery, no flexible generation
                        # C8: Set generation to 0!
                        Cons.create(gP[t] == minZero, "P Gen Min Node " + " at time " + str(t))
                        # C13: Set B to 0
                        Cons.create(B[t] == minZero, "P Gen Min Node " + " at time " + str(t))
            else:
                for t in range(num_time):
                    if gen_i:
                        # If we have a battery
                        # No flex, no uncertainty, thus deterministic cons.
                        # C8&9: Active Power Limits
                        Cons.create(gP[t] >= Pmin, "P Gen Min Node " + " at time " + str(t))
                        Cons.create(gP[t] <= Pmax, "P Gen Max Node " + " at time " + str(t))

                        if t == 0:
                            # C14: SOC at begin and end
                            Cons.create(B[t] == Q_mid - gP[t], "SOC @ Node " + " at time " + str(t))
                        else:
                            # C13: SOC capture
                            Cons.create(B[t] == B[t - 1] - gP[t], "SOC @ Node " + " at time " + str(t))

                        if t == (num_time - 1):
                            # 15: SOC at begin and end
                            Cons.create(B[t] == Q_mid, "End SOC @ Node")

                        # C16,17: SOC limits
                        Cons.create(B[t] >= Q_min, "SOC Min at time " + str(t))
                        Cons.create(B[t] <= Q_max, "SOC Max at time " + str(t))
                    else:
                        # Otherwise, no battery, no flex generation
                        # Otherwise, no battery, no flexible generation
                        # C8: Set generation to 0!
                        Cons.create(gP[t] == minZero, "P Gen Min Node " + " at time " + str(t))
                        # C13: Set B to 0
                        Cons.create(B[t] == minZero, "P Gen Min Node " + " at time " + str(t))

            # Voltage and Flow CC
            if MODE == 3:
                for t in range(num_time):
                    # C6/7: Voltage Limits
                    Cons.create(us[0][t] - z_v * tv[t] >= u_min, "Min U at time " + str(t))
                    Cons.create(us[0][t] + z_v * tv[t] <= u_max, "Max U at time " + str(t))

                    # C12: Flow limits (with gP, just set to 0 if there is none)
                    for la in range(circle_approx):
                        lin_approx = a1[la] * (fP[0][t] + z_f * tf[t]) + a2[la] * fQ[0][t] + a3[la] * fLim
                        Cons.create(lin_approx <= 0, "LA+ " + str(la) + " at time " + str(t))

                        lin_approx = a1[la] * (fP[0][t] - z_f * tf[t]) + a2[la] * fQ[0][t] + a3[la] * fLim
                        Cons.create(lin_approx <= 0, "LA- " + str(la) + " at time " + str(t))

                    # SOC constraints and related constraints
                    # Just because our alpha is 0 does not mean we do not feel the effects of everyone else!
                    # C23: Sum for Voltage Aux
                    rrhou_LHS = sum(
                        [R_inv_dlmp[current - 1][j] * rho_v[j][t] for j in range(nodes - 1)])  # j from 1 to n
                    Cons.create(rrhou_LHS == alpha[current][t], "Sum U on alpha " + str(current) + " at time " + str(t))
                    # C25: Sum for Flow Aux
                    rrhof_LHS = sum([A_inv_dlmp[current - 1][j] * rho_f[j][t] for j in range(nodes - 1)])
                    Cons.create(rrhof_LHS == alpha[current][t], "Sum F on alpha " + str(current) + " at time " + str(t))

                    # C24: SOC for Voltage
                    x_l = R_dlmp[current - 1]
                    x_r = rho_v[current][t] * eT
                    x_lr = (x_l + x_r) @ S_inv_dlmp_list[t]
                    Cons.create(cp.SOC(tf[t], x_lr), "SOC for f at time " + str(t))

                    # C26: SOC for Flow
                    x_l = A_dlmp[current - 1]
                    x_r = rho_f[current][t] * eT
                    x_lr = (x_l + x_r) @ S_inv_dlmp_list[t]
                    Cons.create(cp.SOC(tf[t], x_lr), "SOC for f at time " + str(t))

                    # CX: Aux variables bigger than 0
                    Cons.create(minZero <= tf[t], "min t_f at time " + str(t))
                    Cons.create(minZero <= tv[t], "min t_v at time " + str(t))
                    Cons.create(minZero == rho_v[0][t],
                                "Wired rho_v " + str(t))  # since feeder does not have V or pF/pQ
                    Cons.create(minZero == rho_f[0][t],
                                "Wired rho_f " + str(t))  # since feeder does not have V or pF/pQ

        # Participation Factors: Same For All
        # Use the knowledge that by definition all the alphas must be between 0 and 1
        # Since everyone has to work with this huge block of alphas, this dramatically cuts down the search space
        # it also leaks nothing about anyone else, just a sanity check
        if MODE in [1, 2, 3]:
            for t in range(num_time):
                # Force all between 0 and 1
                for n_alpha in range(nodes):
                    if n_alpha == current and current and not gen_i:
                        # Nodes without battery that are not the feeder
                        Cons.create(alpha[current, t] == minZero,
                                    "Alpha Zero Node " + str(n_alpha) + " at time " + str(t))
                    else:
                        # C27 and 28: Lower and Upper Bounds for Flex Factor (ours is at current!)
                        Cons.create(alpha[n_alpha, t] >= minZero,
                                    "Alpha Min Node " + str(n_alpha) + " at time " + str(t))
                        Cons.create(alpha[n_alpha, t] <= maxOne,
                                    "Alpha Max Node " + str(n_alpha) + " at time " + str(t))

                # C18: Flex Factor Balance
                Cons.create(sum(alpha[:, t]) == maxOne, "Alpha Balance" + " at time " + str(t))

        if MODE == 4:
            for ui in range(num_time):
                for t in range(num_time):
                    # Force all between 0 and 1
                    for n_alpha in range(nodes):
                        if ui:
                            if n_alpha == current and current and not gen_i:
                                # Nodes without battery that are not the feeder
                                Cons.create(alpha[ui, current, t] == minZero,
                                            "Alpha Zero Node " + " at node " + str(ui) + str(
                                                n_alpha) + " at time " + str(
                                                t))
                            else:
                                # C27 and 28: Lower and Upper Bounds for Flex Factor (ours is at current!)
                                Cons.create(alpha[ui, n_alpha, t] >= minZero,
                                            "Alpha Min Node " + " at node " + str(ui) + str(
                                                n_alpha) + " at time " + str(t))
                                Cons.create(alpha[ui, n_alpha, t] <= maxOne,
                                            "Alpha Max Node " + " at node " + str(ui) + str(
                                                n_alpha) + " at time " + str(t))
                        else:
                            # feeder has no uncertainty, so no flex for it!
                            Cons.create(alpha[ui, n_alpha, t] == minZero,
                                        "Alpha Zero Node " + " at node " + str(ui) + str(n_alpha) + " at time " + str(
                                            t))

                    # C18: Flex Factor Balance
                    Cons.create(sum(alpha[ui, :, t]) == maxOne,
                                "Alpha Balance" + " at node " + str(ui) + " at time " + str(t))

        """
        OBJECTIVE
        Include all the multiplier*variables here!
        """
        # Linear cost is just for active power flow injections
        if current == 0:
            cost = sum([l_P[t] * GT[t] - s_P[t] * FT[t] for t in range(num_time)])
        else:
            cost = l_cost * sum(gP)  # Try out

        if MODE == 0 or MODE == 1:
            cost += q_cost * cp.sum_squares(gP)  # C_Q * \sum_t(l_P-s_P)^2 = C_Q * \sum_t(g_P)^2
            cost += q_cost * cp.sum_squares(gP)  # C_Q * \sum_tg_P^2

        if MODE in [2, 3, 4]:
            # Quadratic Cost Constraints for gP and Flex
            # Generation Quadratic Cost
            # C21: SOC constraint for Quadratic generation cost: Just out active generation!
            # Bundle across all timesteps, thus just do it once
            quad_r = gP
            # noinspection PyTypeChecker
            quad_x = C_quad_sqrt @ quad_r
            Cons.create(cp.SOC(quad_genP, quad_x), "Quad Cost P Gen" + " at node " + str(current))

            # Quadratic Terms
            cost += quad_genP

        if MODE == 4:
            # S is the sqrt of alpha times sigma
            # Can be captured as the norm of our alphas (at each uncertain node) times sqrt of Sigma
            for t in range(num_time):
                x_lr = alpha[:, current, t] @ S_inv_dlmp_list[t]
                Cons.create(cp.SOC(S[t], x_lr), "S at time " + str(t))

        """
        Penalty terms as per CVXPY: (rho/2)*sum_squares(local - global - multiplier)    
        This works as sum_{ij}X_{ij}^2 and thus we can have a matrix inside the ()
        Since ECOS is fighting me, accept we are adding (rho/2)*sqrt(sum_squares(local - global - multiplier)) instead
        Should still work but we can now express the second half via a SOC constraint
        Note however that SOC goes per columns, which in our case is per time! This is totally fine, just make aux a vector    

        If we do the other formulation, we need a lambda times residual entry!
        """
        # Active Flows Updates (should be a len(dw_set_i)*num_time matrix)
        pen_cost = rho_2 * cp.sum_squares(fP - global_P + lambda_P)
        # Reactive Flows Updates (should be a len(dw_set_i)*num_time matrix)
        pen_cost += rho_2 * cp.sum_squares(fQ - global_Q + lambda_Q)
        # Voltage Updates (inside should be a len(dw_set_i)*num_time matrix)
        pen_cost += rho_2 * cp.sum_squares(us - global_U + lambda_U)

        if MODE in [1, 2, 3]:
            # Just YOUR quadratic cost for alpha (since remember, we sum up the overall thing!)
            alpha_s_sum = sum([alpha[current, t] * s_list[t] for t in range(num_time)])  # either S or S^2 !!

            # Only if you have alphas
            if gen_i or current == 0:
                cost += q_cost * cp.sum_squares(alpha_s_sum)

            # Alpha Updates (should be a nodes*num_time matrix)
            pen_cost += rho_2 * cp.sum_squares(alpha - global_A + lambda_A)

        if MODE == 3:
            # Rho_f updates (should be a nodes*num_time matrix)
            pen_cost += rho_2 * cp.sum_squares(rho_f - global_RF + lambda_RF)
            # Rho_v updates (should be a nodes*num_time matrix)
            pen_cost += rho_2 * cp.sum_squares(rho_v - global_RV + lambda_RV)

        if MODE == 4:
            # Only if you have alphas
            if gen_i or current == 0:
                cost += q_cost * sum(S)

            # Alphas updates (should be a nodes*nodes*num_time matrix)
            pen_cost += rho_2 * cp.sum_squares(alpha - global_A + lambda_A)

        obj = cp.Minimize(cost + pen_cost)
        prob = cp.Problem(obj, Cons.getAll())

        chk = 1 if prob.is_dcp() else 0
        pipe.send(chk)

        # Primal is local - global of this iteration
        # Dual is global - old global!!!!!!
        primal_residual = cp.sum_squares(fP - global_P) + cp.sum_squares(fQ - global_Q) + cp.sum_squares(us - global_U)
        dual_residual = cp.sum_squares(global_P - global_P_old) + cp.sum_squares(
            global_Q - global_Q_old) + cp.sum_squares(
            global_U - global_U_old)

        if MODE in [1, 2, 3]:
            primal_residual += cp.sum_squares(alpha - global_A)
            dual_residual += cp.sum_squares(global_A - global_A_old)

        if MODE == 3:
            # SOC
            primal_residual += cp.sum_squares(rho_f - global_RF)
            dual_residual += cp.sum_squares(global_RF - global_RF_old)
            primal_residual += cp.sum_squares(rho_v - global_RV)
            dual_residual += cp.sum_squares(global_RF - global_RV_old)

        if MODE == 4:
            primal_residual += cp.sum_squares(alpha - global_A)
            dual_residual += cp.sum_squares(global_A - global_A_old)

        while True:
            # receive iteration
            iteration = pipe.recv()

            if MODE == 0:
                prob.solve(ignore_dpp=True, solver=cp.OSQP)
            elif MODE == 1:
                prob.solve(ignore_dpp=True, solver=cp.OSQP, max_iter=50000)
            elif MODE == 2:
                if current:
                    prob.solve(ignore_dpp=True, solver=cp.ECOS, max_iters=750)
                else:
                    # Solve the LP to get duals
                    prob.solve(ignore_dpp=True, solver=cp.ECOS)
            elif MODE == 3:
                if current:
                    prob.solve(ignore_dpp=True, solver=cp.ECOS, max_iters=750)
                else:
                    # Solve the LP to get duals
                    prob.solve(ignore_dpp=True, solver=cp.ECOS)
            else:
                raise ValueError

            stat = prob.status

            # send in status
            pipe.send(stat)

            vall = prob.value

            # send in values
            pipe.send(vall)
            # Everyone has a coupled variable of fP,fQ,us, so all send in their local values!
            # to reduce storage complexity in controller while still tying time-series to the node it is for
            # we use tuples (node, X[node]) send by current. Thus, this is current's view of node's values for variable X

            fP_send = [(dw_set_i[jP], fP.value[jP, :]) for jP in
                       range(len(dw_set_i))]  # send in D
            fQ_send = [(dw_set_i[jQ], fQ.value[jQ, :]) for jQ in
                       range(len(dw_set_i))]  # send in D
            us_send = [(up_set_i[jU], us.value[jU, :]) for jU in
                       range(len(up_set_i))]  # send in U

            pipe.send(fP_send)
            # Added in: send your GP here!
            pipe.send(gP.value)

            pipe.send(fQ_send)
            pipe.send(us_send)

            # if CC: send in alpha, rho_f, rho_v
            if MODE in [1, 2, 3]:
                # stays the same, just send the whole array
                pipe.send(alpha.value)

            if MODE == 3:
                pipe.send(rho_f.value)
                pipe.send(rho_v.value)

            # We get things back in the correct shape for us locally already!!!
            # So we can just immediately store it!
            global_P.value = pipe.recv()
            global_Q.value = pipe.recv()
            global_U.value = pipe.recv()

            if MODE in [1, 2, 3]:
                global_A.value = pipe.recv()

            if MODE == 3:
                global_RF.value = pipe.recv()
                global_RV.value = pipe.recv()

            if MODE == 4:
                global_A.value = pipe.recv()

            # receive updated global variables and update multipliers while you wait for your next turn
            # updates as follows: local_new - global_new
            # Active Flows Updates (should be a len(up_set_i)*num_time matrix)
            lambda_P.value += fP.value - global_P.value
            # Reactive Flows Updates (should be a len(up_set_i)*num_time matrix)
            lambda_Q.value += fQ.value - global_Q.value
            # Voltage Updates (inside should be a len(dw_set_i)*num_time matrix)
            lambda_U.value += us.value - global_U.value

            if MODE in [1, 2, 3]:
                # Alpha is a nodes*num_time matrix
                lambda_A.value += alpha.value - global_A.value

            if MODE == 3:
                # Rho_f, Rho_v are all nodes*num_time matrices
                lambda_RF.value += rho_f.value - global_RF.value
                lambda_RV.value += rho_v.value - global_RV.value

            if MODE == 4:
                lambda_A.value = alpha.value - global_A.value

            # Take the norm (which is just the Frobenius norm and thus can also handle matrices!) and send to global
            # This should all be floats!
            norm_P = numpy.linalg.norm(fP.value - global_P.value)
            norm_Q = numpy.linalg.norm(fQ.value - global_Q.value)
            norm_U = numpy.linalg.norm(us.value - global_U.value)

            if MODE in [1, 2, 3]:
                norm_A = numpy.linalg.norm(alpha.value - global_A.value)

            if MODE == 3:
                norm_RF = numpy.linalg.norm(rho_f.value - global_RF.value)
                norm_RV = numpy.linalg.norm(rho_v.value - global_RV.value)

            if MODE == 4:
                norm_A = numpy.linalg.norm(alpha.value - global_A.value)

            # First compute then send to not waste time!
            pipe.send(norm_P)
            pipe.send(norm_Q)
            pipe.send(norm_U)

            if MODE in [1, 2, 3]:
                pipe.send(norm_A)
            if MODE == 3:
                pipe.send(norm_RF)
                pipe.send(norm_RV)
            if MODE == 4:
                pipe.send(norm_A)

            r_i = primal_residual.value
            s_i = dual_residual.value

            # send in r_i, s_i for convergence check
            pipe.send(r_i)
            pipe.send(s_i)

            # receive if we terminate or not!
            terminator = pipe.recv()

            if terminator == 1.0:
                # Send in net injection and nodal
                if current == 0:
                    pipe.send(l_P.value)
                    pipe.send(l_Q.value)
                    pipe.send(s_P.value)
                    pipe.send(s_Q.value)

                else:
                    pipe.send(gP.value)
                if gen_i:
                    pipe.send(B.value)

                # Send Shadow Prices
                sp_list = numpy.array(
                    [Cons.getCon("P Balance at Node " + str(current) + " at time " + str(t)).dual_value for t in
                     range(num_time)])

                sp = -1.0 * sp_list
                pipe.send(sp)
                # Send lambda P
                pipe.send(lambda_P.value[0, :])  # just ours

                if MODE in [1, 2, 3]:
                    # Send Flex Prices
                    pi = numpy.array(
                        [Cons.getCon("Alpha Balance" + " at time " + str(t)).dual_value for t in range(num_time)])
                    pipe.send(pi)
                    # Send lambda A
                    pipe.send(lambda_A.value[current, :])  # just ours

                if MODE == 4:
                    # Send Flex Prices
                    pi = numpy.array(
                        [[Cons.getCon("Alpha Balance" + " at node " + str(ui) + " at time " + str(t)).dual_value for t
                          in
                          range(num_time)] for ui in range(nodes)])
                    pipe.send(pi)
                    # Send lambda A
                    pipe.send(lambda_A.value[:, current, :])  # just ours

                # end the loop!
                break

            # Finally, update old global values for new ones
            # Need copy to avoid Python shenanigans
            global_P_old.value = numpy.copy(global_P.value)
            global_Q_old.value = numpy.copy(global_Q.value)
            global_U_old.value = numpy.copy(global_U.value)

            if MODE in [1, 2, 3]:
                global_A_old.value = numpy.copy(global_A.value)

            if MODE == 3:
                global_RF_old.value = numpy.copy(global_RF.value)
                global_RV_old.value = numpy.copy(global_RV.value)

            if MODE == 4:
                global_A_old.value = numpy.copy(global_A.value)
