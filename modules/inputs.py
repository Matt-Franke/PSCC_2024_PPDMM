"""
Here the acquisition, procession and preparation of input data is conducted
"""
import pandas
from currency_converter import CurrencyConverter
import pandas as pd
from datetime import date
import numpy as np
import os

def loading_csv(path: str, ext: bool = False):
    """
    some weird issues with whether it does or does not want a ../ at the start
    :param path:
    :return:
    """
    try:
        if ext:
            df = pd.read_csv(os.path.normpath(path), sep=';', decimal=",")[::-1]

        else:
            df = pd.read_csv(os.path.normpath(path))

    except FileNotFoundError as _:
        new_path = os.path.normpath("../" + path)
        if ext:
            df = pd.read_csv(new_path, sep=';', decimal=",")[::-1]
        else:
            df = pd.read_csv(new_path)

    return df


def get_basic_input():
    """
    This acquires, processes and returns a basic set of pandas dataframes
    """
    c = CurrencyConverter()  # We operate with CHF, but our datasets may not

    # Base unit is kWh (since period is 1 hour) since otherwise the numbers get too small
    # Sizing in real, profiles in pu

    # 1. Load sizings and 2. Load Profiles from CLNR Test Cell 5: Photovoltaic (PV) Base Profile Monitoring
    """
    CLNR Test Cell 5: Photovoltaic (PV) Base Profile Monitoring	
    From https://www.networkrevolution.co.uk/project-library/dataset-tc5-enhanced-profiling-solar-photovoltaic-pv-users/
    TC5.2 Demand Average
    Using Nov 2013 monthly half-hour data which is KW
    Means we need to upsample from 0.5 hrs to 1 hr by getting the average
    And then multiply by 0.001 to go to MW
    """
    fh_ts = pd.date_range('11/1/2013', periods=48, freq='1H')  # generate the timestamps

    loads_df = loading_csv("/data/CLNRL076-TC5/CLNRL076 TC5/TC5_Dem.csv")
    hh_ts = pd.date_range('1/11/2013', periods=48, freq='30T')  # generate the timestamps
    demand_hours = pd.Series(loads_df["Nov 2013"].tolist(), hh_ts).resample(
        "1H").mean().to_frame()  # upsample from 0.5H
    # to H
    demand_hours = demand_hours.multiply(0.001)  # from kW to MW
    demand_hours.columns = ["Demand (MW)"]  # Dataframe to use going forward

    d48: pd.DataFrame = pd.Series(demand_hours["Demand (MW)"].tolist() + demand_hours["Demand (MW)"].tolist(),
                                  fh_ts).to_frame()
    d48.columns = ["Demand (MW)"]

    # 3. PV sizing and # 4. PV profiles
    """
    CLNR Test Cell 5: Photovoltaic (PV) Base Profile Monitoring	
    From https://www.networkrevolution.co.uk/project-library/dataset-tc5-enhanced-profiling-solar-photovoltaic-pv-users/
    Half-hour resolution demand data
    Using Nov 2013 monthly 
    """
    pv_df = loading_csv("data/CLNRL076-TC5/CLNRL076 TC5/TC5_Gen.csv")
    g_l = [max(0.0, i) for i in pv_df["Nov 2013"].tolist()]  # SOmehow there is negative values here?!??!
    gen_hours = pd.Series(g_l, hh_ts).resample("1H").mean().to_frame()  # upsample from 0.5H to H
    gen_hours = gen_hours.multiply(0.001)  # from kWh to MWh
    gen_hours.columns = ["Generation (MWh)"]  # Dataframe to use going forward

    g48: pd.DataFrame = pd.Series(gen_hours["Generation (MWh)"].tolist() + gen_hours["Generation (MWh)"].tolist(),
                                  fh_ts).to_frame()
    g48.columns = ["Generation (MWh)"]
    # 5. Wind power sizing
    # 10 kW each so that the two are roughly equal to all the PV gen
    wind_cap_1 = 10 * 0.001
    wind_cap_2 = 10 * 0.001

    # 6. Wind profile
    """Per Unit Data from https://pureportal.strath.ac.uk/en/datasets/australian-electricity-market-operator-aemo-5
    -minute-wind-power-d 
    Australian Electricity Market Operator (AEMO) 5 Minute Wind Power Data 1st and 2nd Nov 2013 in 5 
    Minute intervals WP 1: Is average of first 11 wind generators (last one is BLUFF1) WP 2: Average of the other 11 wind 
    gens from CLEMGPWF onwards """
    wp_df = loading_csv("data/AEMO/WP.csv")
    fivemin_ts = pd.date_range(start="2013-11-01", end="2013-11-03", freq="5T")
    # Since the above gives us 577 and not 576 timesteps, we add a dummy row that we then delete before resampling

    wp_left: pd.DataFrame = pd.Series(wp_df["Left"].tolist(), fivemin_ts)[:-1].resample("1H").mean().to_frame()
    wp_left.columns = ["Wind Power (p.u.)"]
    wp_left = wp_left.multiply(wind_cap_1)
    wp_left.columns = ["Wind Power (MW) Left"]

    wp_right: pd.DataFrame = pd.Series(wp_df["Right"].tolist(), fivemin_ts)[:-1].resample("1H").mean().to_frame()
    wp_right = wp_right.multiply(wind_cap_2)
    wp_right.columns = ["Wind Power (MW) Right"]

    # 7. Wholesale Market Prices for Day Ahead and Balancing
    """
    From Energinet.dk for Nov 1st and 2nd 2022
    For Spot: DK1 Spot Price
    https://api.energidataservice.dk/dataset/Elspotprices?offset=0&start=2022-11-01T00:00&end=2022-11-04T00:00&filter=%7B%22PriceArea%22:[%22DK1%22]%7D&sort=HourUTC%20DESC&timezone=dk

    For Regulation: DK1 Realtime Market
    https://api.energidataservice.dk/dataset/RealtimeMarket?offset=0&start=2022-11-01T00:00&end=2022-11-04T00:00&filter=%7B%22PriceArea%22:[%22DK1%22]%7D&sort=HourUTC%20DESC&timezone=dk

    Prices are EUR per MWh and we convert to CHF per MWh

    """
    conv = c.convert(1, 'EUR', 'CHF')

    """
    This extracts the Spot price in CHF/kWh for 2 days
    """

    spot_raw_df = loading_csv("data/Energinet/Elspotprices.csv", ext=True)
    spot_raw_df.reset_index(inplace=True, drop=True)
    spot_df = pd.DataFrame({'Timestep': fh_ts, 'Spot Price (CHF/MWh)': spot_raw_df["SpotPriceEUR"].tolist()[1:49]})
    spot_df = spot_df.set_index("Timestep")
    spot_df.index.name = None
    spot_df = spot_df.multiply(conv)

    """
    This extracts the in CHF/kWh for 2 days the Prices for Up and Down Regulation,
    as well as the consumption (i.e. which every reg was not needed)
    We can thus also derive the imbalance signs (-1 for down, 0 for none, 1 for up)
    When Consumption equals to both, we get 0
    Else which ever of the two regs is not equal to consumption, is the imbalance
    """

    reg_raw_df = loading_csv("data/Energinet/RealtimeMarket.csv", ext=True)
    reg_raw_df.reset_index(inplace=True, drop=True)
    reg_df = pd.DataFrame({'Timestep': fh_ts,
                           "Up Reg Price (CHF/MWh)": reg_raw_df["BalancingPowerPriceUpEUR"].tolist()[:48],
                           "Down Reg Price (CHF/MWh)": reg_raw_df["BalancingPowerPriceDownEUR"].tolist()[:48],
                          "Consumption Price (CHF/MWh)": reg_raw_df["BalancingPowerConsumptionPriceEUR"].tolist()[:48]})
    reg_df = reg_df.set_index("Timestep")
    reg_df.index.name = None
    reg_df = reg_df.multiply(conv)
    reg_df.columns = ["Up Reg Price (CHF/MWh)", "Down Reg Price (CHF/MWh)", "Consumption Price (CHF/MWh)"]

    conditions = [
        (reg_df['Up Reg Price (CHF/MWh)'] == reg_df["Consumption Price (CHF/MWh)"])
        & (reg_df["Down Reg Price (CHF/MWh)"] == reg_df["Consumption Price (CHF/MWh)"]),
        (reg_df['Up Reg Price (CHF/MWh)'] == reg_df["Consumption Price (CHF/MWh)"])
        & (reg_df["Down Reg Price (CHF/MWh)"] != reg_df["Consumption Price (CHF/MWh)"]),
        (reg_df['Up Reg Price (CHF/MWh)'] != reg_df["Consumption Price (CHF/MWh)"])
        & (reg_df["Down Reg Price (CHF/MWh)"] == reg_df["Consumption Price (CHF/MWh)"])]
    choices = [0, -1, 1]
    reg_df['Imbalance Sign'] = np.select(conditions, choices)

    conditions = [
        (reg_df['Imbalance Sign'] == 1),
        (reg_df['Imbalance Sign'] <= 0)]
    choices = [reg_df['Up Reg Price (CHF/MWh)'], reg_df["Down Reg Price (CHF/MWh)"]]
    reg_df['Reg Price (CHF/MWh)'] = np.select(conditions, choices)

    # 8. Wholesale Tariff Price
    """
    From Peer-to-peer and community-based markets: A comprehensive review, page 9
    """
    tariff = c.convert(10, 'USD', 'CHF')  # EUR per MWh

    # Export to a CSV!
    joint_df = pd.concat([d48, g48, wp_left, wp_right, spot_df, reg_df], axis=1)
    joint_df = joint_df.rename_axis("Timestep")
    joint_df.to_csv(os.path.normpath("../data/Processed/joint.csv"))

    return demand_hours, gen_hours, wp_left, wp_right, spot_df, reg_df, tariff


def retrieve_basic_inputs():
    """
    Retrieve the already processed data set as a dataframe from a CSV
    :return:
    """
    path = "data/Processed/joint.csv"
    try:
        joint_df = pd.read_csv(path, parse_dates=True)

    except FileNotFoundError as _:
        new_path = "../data/Processed/joint.csv"
        joint_df = pd.read_csv(new_path, parse_dates=True)

    joint_df = joint_df.set_index("Timestep")

    c = CurrencyConverter()  # We operate with CHF, but our datasets may not
    tariff = c.convert(10, 'USD', 'CHF')  # EUR per MWh

    return joint_df, tariff


def get_basic_network():
    # 9. Line Constraints
    """
    Use test case from https://github.com/mieth-robert/DLMP_uncertainty_CodeSupplement/tree/public/data/feeder_data/feeder15
    from the DTU paper  "Distribution Electricity Prices under Uncertainty" by Robert Mieth and Yury Dvorkin.

    This gives us both the connections, and the line susceptances and line limits (its limits on S, but we assume its P)
    (The original is using LinDistFlow not DC-OPF)

    Based on DTU paper  "Distribution Electricity Prices under Uncertainty" by Robert Mieth and Yury Dvorkin.
    So 15 nodes, 8 w/ PV, 2 with wind, 5 with nothing
    """
    lines_df = loading_csv("data/DLMP/lines.csv/lines.csv")
    sources_df = loading_csv("data/DLMP/generators.csv/generators.csv")

    c = CurrencyConverter()  # We operate with CHF, but our datasets may not

    # pf here is tan phi which is constant and equal to Q/P (so pf*P=Q)
    pf = sources_df["q_max"][1] / sources_df["p_max"][1]
    der_cost = c.convert(sources_df["cost"][1], 'USD', 'CHF')

    nodes = 15 + 1
    lines = 14 + 1
    incidence = np.zeros((lines, 2), dtype=int)  # For each branch,  from which node to which
    # ends
    susceptance = np.zeros(lines)  # For each branch, its susceptance
    resistance = np.zeros(lines)
    reactance = np.zeros(lines)
    limits = np.zeros(lines)  # for each branch its max power flow (either direction)

    # connect up the feeder with branch 0 (between nodes 0 and 1)
    branch = 0
    incidence[branch] = [0, 1]
    susceptance[branch] = lines_df["b"][0]
    resistance[branch] = lines_df["r"][0]
    reactance[branch] = lines_df["x"][0]
    limits[branch] = lines_df["s_max"][0]

    for index, row in lines_df.iterrows():
        branch = int(row["index"])
        incidence[branch] = [int(row["from_node"]), int(row["to_node"])]
        susceptance[branch] = row["b"]
        limits[branch] = row["s_max"]
        resistance[branch] = row["r"]
        reactance[branch] = row["x"]

    return incidence, resistance, reactance, susceptance, limits, nodes, lines, pf, der_cost


# noinspection DuplicatedCode
def generate_connect_outflows(nodes: int, lines: int, incidence: list):
    """
    Line flows are denoted by their start and end nodes. Thus when looping over line flows
    it will be useful to fetch relevant values by the nodes and not give the line flow itself an index.
    This is especially true when we flip lines, since this way our ground truth is always
    the start and the end notes.

   Thus we have a node by node matrix connect[from][to] that retrieves the relevant line index
    and a list for each node with their outgoing lineflows

    :param lines: Number of lines
    :param nodes: Number of nodes
    :param incidence: A list of each branch with a list inside with their start and end node
    :param susceptance: Susceptance value for each branch
    :return: A n*n matrix of susceptance values and a list for each nodes of their outflows
    """
    # check shapes first of all
    if len(incidence) != lines:
        raise ValueError

    outflows = [[] for _ in range(nodes)]
    connect = [[-1 for _ in range(nodes)] for _ in range(nodes)]  # if -1, no line between the two

    # NOTE: WE ASSUME LINES ARE STORED IN THE SAME ORDER
    # IN BOTH INCIDENCE AND SUSCEPTANCE!!!!

    for idx, b in enumerate(incidence):
        f = int(b[0])
        t = int(b[1])
        outflows[f].append([f, t])
        outflows[t].append([t, f])

        connect[f][t] = idx
        connect[t][f] = idx

    return connect, outflows



