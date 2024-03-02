"""
Here the overall scenario is set up and updated
"""
import pandas


class Community:
    """
    Defines the overall community, bundles actors, contains power network
    """

    def __init__(self, nodes: int, lines: int, node_split: dict, period: float, num_time: int, MODE: int):
        """
        Setup up high level view of the community
        :param nodes: Number of nodes inside
        :param lines: Number of lines between nodes
        :param pv: Nodes that are pv
        :param wind: Nodes that have wind
        :param load: Nodes are just loads
        :param period: how many hours per timesteps
        :return:
        """
        # Note we assume a node has either pv or wind at this point
        self.nodes = nodes
        # plus feeder at node 0 that connects that what was otherwise the first node
        self.lines = lines
        self.node_split = node_split

        # Feeder at nodes 0
        # Wind at nodes 7 and 12
        # Pure loads at 5, 9, 10, 14, 15
        # PV at 1, 2, 3, 4, 6, 8, 11, 13
        self.feeder = 0
        self.parties = [i for i in range(nodes) if i != self.feeder]  # All but root
        self.pv = [1, 2, 3, 4, 6, 8, 11, 13]
        self.load = [5, 9, 10, 14, 15]
        self.wind = [7, 12]
        combo = self.wind + self.pv
        combo.sort()
        self.gens = combo
        self.period = period
        self.num_time = num_time  # number of timesteps
        self.CC = True if MODE else False
        self.MODE = MODE

    def setup(self, gen_char: dict, stor_char: dict, lines_char: dict, net_char: dict, cost_char: dict):
        """
        Takes in lists of characters of demand, generation and storage
        :param gen_char:
        :param stor_char:
        :param lines_char:
        :param net_char:
        :param cost_char:
        """
        #####
        # Batteries
        #####
        self.Q_min = [stor_char["Lo"] if i in self.gens else 0.0 for i in range(self.nodes)]
        self.Q_max = [stor_char["Hi"] if i in self.gens else 0.0 for i in range(self.nodes)]
        self.Q = [stor_char["Cap"] if i in self.gens else 0.0 for i in range(self.nodes)]
        self.P_max = [stor_char["Max"] if i in self.gens else 0.0 for i in range(self.nodes)]
        Q_mdpt = (stor_char["Hi"] + stor_char["Lo"]) / 2.0
        self.Q_midnight = [Q_mdpt if i in self.gens else 0.0 for i in range(self.nodes)]

        for i in self.pv:
            self.Q[i] *= stor_char["PV"]

        for i in self.wind:
            self.Q[i] *= stor_char["Wind"]

        #####
        # Gen and Demand
        #####
        self.pf = gen_char["PF"]
        self.max_ap = gen_char["AP"]
        self.Q_batt = gen_char["Q"]
        self.g_P_max = [0.0 for _ in range(self.nodes)]
        self.g_P_min = [0.0 for _ in range(self.nodes)]
        self.g_Q_max = [0.0 for _ in range(self.nodes)]
        self.g_Q_min = [0.0 for _ in range(self.nodes)]

        # Batteries are the actual flexible generation at the DER nodes!
        for gg in self.gens:
            # noinspection PyTypeChecker
            self.g_P_min[gg] = -1.0 * self.P_max[gg]  # -1*P_c_max
            self.g_P_max[gg] = self.P_max[gg]  # P_d_max

        #####
        # Cost and Utility
        #####
        self.utility = [cost_char["U_P"] if i != self.feeder else 0.0 for i in range(self.nodes)]  # for all in P
        self.lin_cost = [cost_char["L_P"] if i != self.feeder else 0.0 for i in range(self.nodes)]
        self.quad_cost = [cost_char["Q_P"] if i != self.feeder else cost_char["Q_F"] for i in range(self.nodes)]
        self.tariff = cost_char["T"]

        #####
        # Lines and Network
        #####
        # For lines, enter the value for resistance, reactance and limit to the receiving node
        # taken from perspective of C.feeder
        # thus all nodes have an entry besides C.feeder which  is set to 0 and skipped
        self.ancestor = net_char["A"]
        self.children = net_char["C"]
        self.tree_order = net_char["TO"]
        self.connect = net_char["CN"]
        self.outflows = net_char["OF"]

        self.r = [0.0 for _ in range(self.nodes)]
        self.x = [0.0 for _ in range(self.nodes)]
        self.S_max = [0.0 for _ in range(self.nodes)]

        for nn in self.tree_order:
            # update your children
            for chi in self.children[nn]:
                li = self.connect[nn][chi]  # the line that connects the two of you
                self.r[chi] = lines_char["R"][li]
                self.x[chi] = lines_char["X"][li]
                self.S_max[chi] = lines_char["S"][li]

    def forecasts(self, load, pv, wind1, wind2, spot, reg, st_dev):
        """
        Imports Day Ahead Forecasts for fixed demand, PV and Wind (sep for both locations)
        And Wholesale Spot and Regulation Prices
        """
        self.forecast = {}

        # Net Demand
        self.forecast["dP"] = [[] for _ in range(self.num_time)]
        self.forecast["dQ"] = [[] for _ in range(self.num_time)]
        self.load = load
        for t in range(self.num_time):
            self.forecast["dP"][t] = [abs(load[t]) if i != self.feeder else 0.0 for i in range(self.nodes)]  # for all in P
            self.forecast["dQ"][t] = [abs(self.pf*load[t]) if i != self.feeder else 0.0 for i in range(self.nodes)]  # for all in P

        # St Devs
        if self.CC:
            self.forecast["SD"] = st_dev

        # Prices
        self.forecast["FT"] = spot - self.tariff
        self.forecast["GT"] = spot + self.tariff
        self.forecast["WM"] = spot
        self.forecast["BM"] = reg

        # PV and Wind
        self.forecast["PV"] = pv
        self.forecast["Wind1"] = wind1
        self.forecast["Wind2"] = wind2
