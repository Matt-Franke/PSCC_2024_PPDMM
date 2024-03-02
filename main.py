"""
This is the main code file for the master thesis,
in charge of orchestrating the overall functionality while calling from
the other files to get functions and further functionality

Master Thesis Project: Matthias Franke (ETH), 2022-2023
High Level Flow:

0. GATHER INPUTS
1. SETUP THE PROTOCOL
2. SCENARIO CREATION
3. GENERATE AND COMBINE FORECASTS
4. JOINT MARKET CLEARING
5. GATHER IMBALANCE AND ACTIVATE RESERVES
6. GATHER AND RECOVER MEASUREMENTS
7. IMBALANCE SETTLEMENT PROCESS
8. UPDATE COVARIANCE MATRIX
9. THE END

"""
import time
from modules import dayahead, inputs, trivia, scenario
from runtime import security
import numpy
import random
import math

"""
0. GATHER INPUTS
Need:
1. Load sizings
2. Load Profiles
3. PV sizing
4. PV profiles
5. Wind power sizing
6. Wind profile
7. Wholesale Market Prices for Day Ahead and Balancing
8. Wholesale Tariff Price
9. Line Constraints
"""
start_prot = time.time()
# Level 0: Import some basic datasets
joint_df, tariff = inputs.retrieve_basic_inputs()
incidence, resistance, reactance, susceptance, limits, nodes, lines, pf, der_cost = inputs.get_basic_network()
connect, outflows = inputs.generate_connect_outflows(nodes, lines, incidence)
ancestor, children, tree_order = trivia.get_ancestry(nodes, outflows, 0)
net_char = {"A": ancestor, "C": children, "TO": tree_order, "CN": connect, "OF": outflows}
print("Data Import Complete")
SOLVERS = ["CENTRAL", "NORMAL", "SECURE"]
"""
1. SETUP THE PROTOCOL

Define actors, security setting, network
"""
PLOT = False
MODE = 0  # Which Mode we are running?
SCENARIO = 0
MULTI = 0.0  # 0, 0.5 or 1.5
BAL = 0  # leave it like this so that we get all the values
SOLVER = SOLVERS[0]
SECURE = False  # whether to run the secure formulation or not
COV = False

print("Mode {}, Scenario {}, Multi {}".format(MODE, SCENARIO, MULTI))

days = 1
num_time = 24 * days
period = 1.0  # hour
node_split = {"pv": 8, "wind": 2, "load": 5}

C = scenario.Community(nodes, lines, node_split, period, num_time, MODE)  # Create a community with network inside

# units of power are MW
# period is 1-hour
# costs are in CHF per MWh
# Remember, everything is in MW and MWh, so data sources may need to be converted!!
"""
ESTABLISH BASIC FACTORS
SWAP OUT DEPENDING ON TEST SCENARIO
"""
der_factor = 1.0
sd_factor = 1.0
batt_factor = 1.0
dem_factor = 1.0

if SCENARIO == 1:
    # Switch to DER
    der_factor = MULTI
elif SCENARIO == 2:
    # Switch to SD
    sd_factor = MULTI
elif SCENARIO == 3:
    # Switch to BATTERY
    batt_factor = MULTI
elif SCENARIO == 4:
    # Switch to DEMAND
    dem_factor = MULTI
else:
    # Baseline
    pass

"""
2. SCENARIO CREATION

Define actors, forecasts, network
"""
"""
FLEXIBLE GENERATORS AND DEMAND
Done exclusivel y via batteries. Thus here we only choose if they can do reactive generation
The Max Apparent Power of the inverters (same for all)
The power factor for demand

"""
# pf
g_Q_batt = False  # Can Battery Produce Reactive Power or Note
max_ap = 1.0  # MVA: Fake Value at the moment
gen_char = {"Q": g_Q_batt, "PF": pf, "AP": max_ap}

"""
BATTERIES
Each Battery has a capacity 20 kWH
And min and max SOC (percentages) Q_min, Q_max of 20% and 80%
Each Battery has a maximum charging and discharging power P_c_max, P_d_max of 10 kW each
Wind gets 3 batteries per location, PV gets 1
This only changes Q
"""
SOC_lo = 0.2
SOC_hi = 0.8
SOC_cap = 0.02 * batt_factor
SOC_max = 0.01
stor_char = {"Lo": SOC_lo, "Hi": SOC_hi, "Cap": SOC_cap, "Max": SOC_max, "PV": 1, "Wind": 3}

"""
LINES
Each line has resistance, reactance and an apparent power limit
We want entry i in this to refer to both Line i and Node i
And thus we use a mapping in case there was some weird turning in the input
"""
# Since R and X are in OHMS, they do not need to scale
# S however does
lines_char = {"R": resistance, "X": reactance, "S": 2.0 * limits}

"""
Cost
"""
util_parties = 0.0
quad_parties = der_cost / 100000.0
quad_feeder = quad_parties  # idk just something larger
cost_char = {"U_P": util_parties, "L_P": der_cost, "Q_P": quad_parties, "Q_F": quad_feeder, "T": tariff}

"""
IMPORT INTO COMMUNITY!
"""
C.setup(gen_char, stor_char, lines_char, net_char, cost_char)

print("Setup of Network Complete")

"""
3. GENERATE AND COMBINE FORECASTS

- For DERs and Demand
- For Uncertainty
- For Prices
"""
pv_forecast = der_factor * 2 * joint_df["Generation (MWh)"].values[:24 * days]
dem_forecast = dem_factor * joint_df["Demand (MW)"].values[:24 * days]
wp1_forecast = der_factor * 1.5 * joint_df["Wind Power (MW) Left"].values[:24 * days]
wp2_forecast = der_factor * 1.5 * joint_df["Wind Power (MW) Right"].values[:24 * days]


sd = 0.2 * sd_factor
st_dev_forecast = numpy.array(
    [[dem_forecast[t] * sd if i != C.feeder else 0.0 for i in range(C.nodes)] for t in range(C.num_time)],
    dtype=object)

spots_forecast = joint_df["Spot Price (CHF/MWh)"].values[:num_time]
reg_forecast = joint_df["Reg Price (CHF/MWh)"].values[:num_time]
up_reg = joint_df["Up Reg Price (CHF/MWh)"].values[:num_time]
dw_reg = joint_df["Down Reg Price (CHF/MWh)"].values[:num_time]


C.forecasts(dem_forecast, pv_forecast, wp1_forecast, wp2_forecast, spots_forecast, reg_forecast, st_dev_forecast)


"""
4. JOINT MARKET CLEARING

CENTRAL: solve market via central solver
NORMAL:  solve market via insecure ADMM
SECURE:  solve market via secure ADMM
"""


BC = False  # Whether to retrieve Binding Constraints of Flows
SAVE = True
CC = True if MODE else False

# Run auction to get # P Gen, P Dem, P Price, A, A Price, Balances
path = "data/Processed/results_CENT" + str(MODE) + "_DATA_" + str(SCENARIO) + "_" + str(int(2 * MULTI)) + ".csv"
MCP = dayahead.Auction(C)

if SOLVER == "NORMAL":
    schedules, demand, lambdas, alphas, pies, balances, res_stat, results_df = security.run_ADMM(MODE, SCENARIO, MULTI, BAL)
    MCP.set_df(results_df)
elif SOLVER == "SECURE":
    schedules, demand, lambdas, alphas, pies, balances, res_stat, results_df = security.run_SEC_ADMM(MODE, SCENARIO, MULTI, BAL)
    MCP.set_df(results_df)
    security.run_dvs(nodes)
else:
    schedules, demand, lambdas, alphas, pies, balances, res_stat = MCP.run_simulation(BC)
    MCP.export(path)
    print("Results Saved")

if PLOT:
    trivia.plot_results(path, CC, num_time)
    print("Results Graphed")

print("Mode {}, Scenario {}, Multi {}".format(MODE, SCENARIO, MULTI))
print(res_stat)

"""
5. GATHER IMBALANCE AND ACTIVATE RESERVES

This is purely simulating an error scenario and then triggering the already schedule part. factors
"""

if CC:
    sd = 0.05  # Experienced uncertainty!
else:
    sd = 0.0

deltas_i = MCP.get_actuals(dem_forecast, sd)

for t in range(num_time):
    deltas_i[t][MCP.C.feeder] = 0

delta_total = [sum(deltas_i[t]) for t in range(num_time)]
print("Total Errors Determined:")
print(delta_total)
# Here you then activate the reserves and do a final diff to find how much you are buying at regulation rate

schedules_up, demand_up = MCP.update_flows(schedules, demand, alphas, deltas_i, delta_total)
print("Response Policies Activated")
print("Updated Power Flows according to Schedules")

"""
6. GATHER AND RECOVER MEASUREMENTS

Generate final measurement deviation, gather measurements for honest, recover measurements for liars
"""

injections = []  # injection[t][i] is the active power net injection

# Let the liars be all the prime numbered nodes
# Out of 16 nodes, this gives us 7 liars and nicely distributed through the network
# We shall thus do all timesteps at once
liars = [2, 3, 5, 7, 9, 11, 13]
honest = [i for i in range(nodes) if i not in liars]

if SECURE:
    netP = numpy.zeros((nodes, num_time))
    lineP = numpy.zeros((nodes, num_time))
    ni_sched = numpy.zeros((nodes, num_time))
    # Fill in actual values for honest and 0s for liars
    for t in range(num_time):
        known_values = MCP.get_known(honest, t)  # applies error one more time
        for i in range(nodes):
            if i in honest:
                netP[i, t] = known_values[i]["p"]
                lineP[i, t] = known_values[i]["P"]

            ni_sched[i,t] = schedules_up[i][t] - demand[i][t]

    success = security.update_states(netP.tolist(), lineP.tolist(), delta_total, ni_sched.tolist(), nodes)
    injections = security.run_recover(honest, liars, nodes)
else:
    netP = numpy.zeros((num_time, nodes))
    for t in range(num_time):
        known_values = MCP.get_known(honest, t)
        unknown_values = MCP.recover_geo(honest, known_values)
        for i in range(nodes):
            if i not in honest:
                known_values[i] = unknown_values[i]
            netP[t, i] = known_values[i]["p"]

    injections = netP.tolist()

print("Recovered Injection Measurements for all Nodes and Timesteps")

"""
7. IMBALANCE SETTLEMENT PROCESS

Simulate the DSO balancing out any remaining deviations, including those caused from the recovery process,
and produce the final financial balances for the day
"""

# If you contribute to the problem, you pay whichever regulation is needed
# If you are helping, then you get paid at the wholesale spot price

up_reg = joint_df["Up Reg Price (CHF/MWh)"].values[:num_time]
dw_reg = joint_df["Down Reg Price (CHF/MWh)"].values[:num_time]
spot_ws = joint_df["Spot Price (CHF/MWh)"].values[:num_time]

if SECURE:
    balances, dvs = security.run_imb(nodes, up_reg.tolist(), dw_reg.tolist(), spot_ws.tolist(), tariff, MODE)
    print("DVS check returned {}".format(dvs))
    print("Determined Penalties")

    print("Finalized Balances")

    for i in range(nodes):
        print("Balance Node " + str(i) + ": " + str(balances[i]) + " CHF")

    balance_budget = sum(balances)
    print(balance_budget)
    print("Success")

else:
    deltas_actual_i = [[0 for _ in range(nodes)] for _ in range(num_time)]
    for ttt in range(num_time):
        for iii in range(nodes):
            if not iii:
                deltas_actual_i[ttt][iii] = 0
            else:
                da = injections[ttt][iii]  # Actual net injection of P
                ds = schedules_up[iii][ttt] - demand[iii][ttt]  # gP + netdemand of MPC - alpha * Delta
                # if positive, oversupply, if negative undersupply
                deltas_actual_i[ttt][iii] = da - ds

    delta_actual_total = [sum(deltas_actual_i[t]) for t in range(num_time)]
    imb_penalties = MCP.penalty(deltas_actual_i, delta_actual_total, up_reg, dw_reg, spot_ws)

    print("Determined Penalties")
    print(imb_penalties)
    print("Determined Final Regulation Payments")

    for i in range(nodes):
        for t in range(days * 24):
            # Day Ahead
            # Feeder already done in auction
            if i != MCP.C.feeder:
                imba_pen = imb_penalties[t][i]
                balances[i][t] += imba_pen  # Penalty paid by responsible
                balances[MCP.C.feeder][t] -= imba_pen  # Penalty paid to Feeder

    print("Finalized Balances")

    final_balance = []
    for i in range(nodes):
        final_balance.append(sum([balances[i][t] for t in range(num_time)]))

    for i in range(nodes):
        print("Balance Node " + str(i) + ": " + str(final_balance[i]) + " CHF")

    balance_budget = sum(final_balance)
    print(balance_budget)
    print("Success")

"""
8. UPDATE COVARIANCE MATRIX
Works with fake values, since it is merely a proof of concept.
Without actual historic data, this would not produce useful values anyways
"""

if SECURE and COV:
    random_flt = numpy.random.rand(nodes, num_time)
    random_int = numpy.random.randint(-5, 5, (nodes, num_time))
    R = random_flt+random_int
    S, s = security.run_cov(nodes, num_time, R.tolist())


"""
9. THE END
"""
end_prot = time.time()
tot_prot = (end_prot - start_prot) / 60
print("Mode {}, Scenario {}, Multi {}".format(MODE, SCENARIO, MULTI))
print("Took {} minutes".format(tot_prot))