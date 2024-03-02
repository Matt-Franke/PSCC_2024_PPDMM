"""
Here the privacy-preserving features are coordinated and ran
"""
import subprocess
import sys
import time
import datetime
import json
import numpy
import pandas
import os


def run_ADMM(MODE, SCENARIO, MULTI, BAL):
    """
    Calls ADMM.py with the specified parameters and returns the results
    We call this via subprocess, which while not ideal practice, removes the worry about namespaces
    :param MODE: betwen 0 and 3
    :param SCENARIO: 0 through 6, though 1-5 is the sensitivty testing
    :param MULTI: 0.0, 0.5 or 1.5 (0.1 for Scenario 4 instead of 0.0)
    :param BAL: 0 or 1
    :return:
    """
    endings = ["DET", "GEN_LIN", "GEN_SOC", "FULL"]
    SOLVER = os.path.normpath("./runtime/ADMM.py")
    path = os.path.normpath("./runtime/temp/results_" + endings[MODE] + "_DATA_" + str(SCENARIO) + "_" + str(
        int(2 * MULTI)) + ".txt")

    command = [sys.executable, SOLVER, str(MODE), str(SCENARIO), str(MULTI), str(BAL)]
    now = datetime.datetime.now()
    print("Begin mode {} with scenario {} with Multi {} at {}".format(endings[MODE], SCENARIO, MULTI, now))
    st = time.time()
    cmd = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = cmd.communicate()
    et = time.time()

    # Works when run but not when debugged
    if cmd.returncode == 0:
        print("Took {} minutes".format((et - st) / 60))
    else:
        print(err)
        raise ValueError

    with open(path, 'w') as f:
        f.write(out)

    # Read in the json file and keep as lists of lists!
    with open(os.path.normpath("./runtime/temp/auction.json"), 'r') as openfile:
        # Reading from json file
        input_dict = json.load(openfile)
    schedules = input_dict["schedules"]
    demand = input_dict["demand"]
    lambdas = input_dict["lambdas"]
    alphas = input_dict["alphas"]
    pies = input_dict["pies"]
    balances = input_dict["balances"]
    res_stat = input_dict["res_stat"]

    # Recover the results dataframe to hand to the simulator
    path = os.path.normpath("data/Processed/results_" + endings[MODE] + "_DATA_" + str(SCENARIO) + "_" + str(int(2 * MULTI)) + ".csv")
    joint_df = pandas.read_csv(path)

    return schedules, demand, lambdas, alphas, pies, balances, res_stat, joint_df


def run_SEC_ADMM(MODE, SCENARIO, MULTI, BAL):
    """
    Calls SEC_ADMM.py with the specified parameters and returns the results
    We call this via subprocess, which while not ideal practice, removes the worry about namespaces
    :param MODE: betwen 0 and 3
    :param SCENARIO: 0 through 6, though 1-5 is the sensitivty testing
    :param MULTI: 0.0, 0.5 or 1.5 (0.1 for Scenario 4 instead of 0.0)
    :param BAL: 0 or 1
    :return: schedules, demand, lambdas, alphas, pies, balances, res_stat
    """
    endings = ["SEC_DET", "SEC_GEN_LIN", "SEC_GEN_SOC", "SEC_FULL"]
    SOLVER = os.path.normpath("./runtime/SEC_ADMM.py")
    path = os.path.normpath("./runtime/temp/results_" + endings[MODE] + "_DATA_" + str(SCENARIO) + "_" + str(
        int(2 * MULTI)) + ".txt")

    command = [sys.executable, SOLVER, str(MODE), str(SCENARIO), str(MULTI), str(BAL)]
    now = datetime.datetime.now()
    print("Begin mode {} with scenario {} with Multi {} at {}".format(endings[MODE], SCENARIO, MULTI, now))
    st = time.time()
    cmd = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = cmd.communicate()
    et = time.time()

    # Works when run but not when debugged
    if cmd.returncode == 0:
        print("Took {} minutes".format((et - st) / 60))
    else:
        print(err)
        raise ValueError

    with open(path, 'w') as f:
        f.write(out)

    # Read in the json file and keep as lists of lists!
    with open(os.path.normpath("./runtime/temp/auction.json"), 'r') as openfile:
        # Reading from json file
        input_dict = json.load(openfile)
    schedules = input_dict["schedules"]
    demand = input_dict["demand"]
    lambdas = input_dict["lambdas"]
    alphas = input_dict["alphas"]
    pies = input_dict["pies"]
    balances = input_dict["balances"]
    res_stat = input_dict["res_stat"]

    # Recover the results dataframe to hand to the simulator
    path = os.path.normpath(
        "data/Processed/results_" + endings[MODE] + "_DATA_" + str(SCENARIO) + "_" + str(int(2 * MULTI)) + ".csv")
    joint_df = pandas.read_csv(path)

    return schedules, demand, lambdas, alphas, pies, balances, res_stat, joint_df


def update_states(netP, line_flow, delta_total, ni_sched, nodes):
    """
    We wish to take the actual line flow P and nodal net injection p and the final error
    and add it to the stored state for further usage
    :param line_flow: active power line flow for each node
    :param nodes: the number of nodes to allow for paging the state files
    :param netP: [n][t] LoL of actual active power net injections
    :param delta_total: [t] list of Big Delta, the final forecast of the error
    :return: return code
    """
    for i in range(nodes):
        path_state = os.path.normpath("./runtime/states/" + str(i) + ".json")

        # Retrieve the dictionay
        with open(path_state, 'r') as openfile:
            # Reading from json file
            state_dict = json.load(openfile)

        # Add ni_act, fP_act, Delta into the state file
        state_dict["ni_act"] = netP[i]
        state_dict["ni_sched"] = ni_sched[i]
        state_dict["fP_act"] = line_flow[i]
        state_dict["Delta"] = delta_total

        # Write it back
        result_string = json.dumps(state_dict)
        with open(path_state, "w") as outfile:
            outfile.write(result_string)

    return True


def run_recover(honest, liars, nodes):
    """
    Takes the Operating Time results of honest parties
    and recovers deterministically via SMPC the net injectios of all, including liars
    :param nodes: num of nodes
    :param honest: list of reporting nodes
    :param liars:  list of not reporting nodes
    :return: [t][i] net injections
    """
    # Store honest and dishonest nodes
    iff_dict = {"honest": honest, "liars": liars}
    result_string = json.dumps(iff_dict)
    with open(os.path.normpath("./runtime/temp/iff.json"), "w") as outfile:
        outfile.write(result_string)

    # RUN SMPC
    command_recov = [sys.executable, os.path.normpath("./runtime/recover.py"), "-M" + str(nodes), "--no-log"]
    path_recov = os.path.normpath("./runtime/temp/recov.json")

    lt = time.time()
    timing = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(lt))
    print("At time: " + timing)
    print("Begin SMPC Recover")
    tic = time.time()
    cmd = subprocess.Popen(command_recov, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, err = cmd.communicate()
    if cmd.returncode != 0:
        print(err)
        raise ValueError
    toc = time.time()
    print("SMPC Recover took {} seconds or {} minutes".format(toc - tic, (toc - tic) / 60))

    # Read the json file:
    with open(path_recov, 'r') as openfile:
        # Reading from json file
        result_dict = json.load(openfile)
    netP_act = result_dict["p"]

    # Update the state files!
    for i in range(nodes):
        path_state = os.path.normpath("./runtime/states/" + str(i) + ".json")
        # Retrieve the dictionay
        with open(path_state, 'r') as openfile:
            # Reading from json file
            state_dict = json.load(openfile)

        # Update ni_act in the state file
        state_dict["ni_act"] = netP_act[i]

        # Write it back
        result_string = json.dumps(state_dict)
        with open(path_state, "w") as outfile:
            outfile.write(result_string)

    # Transpose netP_act to get [t][i] injections
    inj = numpy.array(netP_act)
    inj = numpy.transpose(inj)
    injections = inj.tolist()

    return injections


def run_dvs(nodes):
    # RUN SMPC
    command_dvs = [sys.executable, os.path.normpath("./runtime/dvs.py"), "-M" + str(nodes), "--no-log"]

    lt = time.time()
    timing = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(lt))
    print("At time: " + timing)
    print("Begin SMPC DVS Phase 1")
    tic = time.time()
    cmd = subprocess.Popen(command_dvs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, err = cmd.communicate()
    if cmd.returncode != 0:
        print(err)
        raise ValueError
    toc = time.time()
    print("SMPC DVS Phase 1 took {} seconds or {} minutes".format(toc - tic, (toc - tic) / 60))


def run_imb(nodes, up_reg, dw_reg, spot_ws, tariff, MODE):
    """
    Runs the SMPC imbalance script with the update state files
    Calculates difference between actual and scheduled net injections
    Loads in regulation prices and calculates the imbalance update
    NOTE: WE ASSUME THE SECURE BALANCE WAS RUN PRIOR TO THIS!
    :param tariff: the fixed tariff for the feeder
    :param up_reg: the up regulation price on WS market
    :param dw_reg: the down regulation price on WS market
    :param spot_ws: the spot price on WS market
    :param nodes: num of nodes
    :return: [n][t] balances
    """
    # Store prices
    imb_dict = {"up": up_reg, "dw": dw_reg, "spot": spot_ws, "tariff": tariff, "MODE": MODE}
    result_string = json.dumps(imb_dict)
    with open(os.path.normpath("./runtime/temp/imb_prices.json"), "w") as outfile:
        outfile.write(result_string)

    # RUN SMPC
    command_imb = [sys.executable, os.path.normpath("./runtime/settlement.py"), "-M" + str(nodes), "--no-log"]
    path_imb = os.path.normpath("./runtime/temp/imb.json")

    lt = time.time()
    timing = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(lt))
    print("At time: " + timing)
    print("Begin SMPC Imbalance")
    tic = time.time()
    cmd = subprocess.Popen(command_imb, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, err = cmd.communicate()
    if cmd.returncode != 0:
        print(err)
        raise ValueError
    toc = time.time()
    print("SMPC Imbalance took {} seconds or {} minutes".format(toc - tic, (toc - tic) / 60))

    # Read the json file:
    with open(path_imb, 'r') as openfile:
        # Reading from json file
        result_dict = json.load(openfile)
    balances = result_dict["balances"]
    DVS = result_dict["DVS"]

    return balances, DVS


def run_cov(nodes, num_time, R):
    """
    Taking a fake n*t matrix, we will produce the resulting covariance matrix S
    and the sqrt of sum of variances s
    :param num_time:
    :param nodes:
    :param R:
    :return: S,s
    """
    # Store prices
    for i in range(nodes):
        path_state = os.path.normpath("./runtime/temp/cov_" + str(i) + ".json")

        state_dict = {"vals": R[i], "num_time": num_time}

        # Write it back
        result_string = json.dumps(state_dict)
        with open(path_state, "w") as outfile:
            outfile.write(result_string)

    # RUN SMPC
    command_cov = [sys.executable, os.path.normpath("./runtime/cov.py"), "-M" + str(nodes), "--no-log"]
    path_cov = os.path.normpath("./runtime/temp/cov.json")

    lt = time.time()
    timing = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(lt))
    print("At time: " + timing)
    print("Begin SMPC Covariance")
    tic = time.time()
    cmd = subprocess.Popen(command_cov, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, err = cmd.communicate()
    if cmd.returncode != 0:
        print(err)
        raise ValueError
    toc = time.time()
    print("SMPC Covariance took {} seconds or {} minutes".format(toc - tic, (toc - tic) / 60))

    # Read the json file:
    with open(path_cov, 'r') as openfile:
        # Reading from json file
        result_dict = json.load(openfile)
    S = result_dict["S"]
    s = result_dict["s"]

    return S, s
