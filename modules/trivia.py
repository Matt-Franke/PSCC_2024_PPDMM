"""
Here are all class and function that are useful but not with a specific category
"""
import pandas
from numpy.random import default_rng
from bokeh.plotting import figure, show, save, output_file
from bokeh.resources import CDN
from bokeh.models import Legend
from bokeh.palettes import Category20
import itertools
from numpy.linalg import norm
from numpy import array
import math


def next_beta(old_beta: float) -> float:
    inside = 1.0 + 4.0 * math.pow(old_beta, 2.0)
    return (1.0 + math.sqrt(inside)) / 2.0


def plot_residuals(residuals: array, title: str, max_its: int):
    """
    Takes in an array of residuals and plots each row (i.e. for each node)
    """
    output_file(filename=".\\plots\\" + title + ".html", title=title)
    p1 = figure(title=title + " Multiplier Norm", x_axis_label="Iterations",
                y_axis_label="Norm across timesteps of Multiplier",
                toolbar_location="above")
    its = []
    c_pol = "mute"

    colours = colour_gen()
    ts1 = [j for j in range(max_its + 1)]
    ct1 = 0

    for col, colour in zip(residuals, colours):
        # col is the name of the column in question
        r = p1.line(ts1, col, line_width=2, alpha=0.8, color=colour)
        its.append((title + " " + str(ct1), [r]))
        ct1 += 1

    legend = Legend(items=its, location="center", click_policy=c_pol)
    p1.add_layout(legend, 'right')
    show(p1)


def norm_across_time(residual: list):
    """
    Take the Euclidean norm over a list and return the float result
    """
    rezzie = norm(array(residual))

    return rezzie


def colour_gen():
    """"
    As per the comment from user Elliot on StackOverflow at
    https://stackoverflow.com/questions/39839409/when-plotting-with-bokeh-how-do-you-automatically-cycle-through-a-color-pallett
    """
    palette = Category20[20]  # or another
    colours = itertools.cycle(palette)
    return colours


def gen_plot(df: pandas.DataFrame, ts: pandas.Series, title: str, x: str, y: str, mute_it: bool = True,
             save_it: bool = False):
    """
    Generates a Bokeh Plot from a Dataframe
    :param df: the input, which requires Timestep to be included for the timeseries plotting
    :param ts: the timeseries
    :param title: Title of Graph
    :param x: X Axis Title
    :param y: Y axis Title
    :param mute_it: If true, mute, else hide
    :param save_it: if it should be saved to a file titled after the title of axis
    :return: a Bokeh plot object which can then be
    """
    p = figure(title=title, x_axis_label=x, y_axis_label=y, toolbar_location="above")
    its = []
    c_pol = "mute" if mute_it else "hide"

    colours = colour_gen()

    for col, colour in zip(df, colours):
        # col is the name of the column in question
        r = p.line(ts, df[col], line_width=2, alpha=0.8, color=colour)
        its.append((col, [r]))
    legend = Legend(items=its, location="center", click_policy=c_pol)
    p.add_layout(legend, 'right')

    if save_it:
        new_title = title.replace(" ", "_") + ".html"
        tt = save(obj=p, filename=new_title, resources=CDN, title=title)
        print(tt)  # the location it was saved to
    return p


def plot_results(path: str, CC: bool, num_time: int = 24, NODAL: bool = False):
    """
    CC==False:
    1. Price for DA at Feeder relative to FT and GT
    2. Supply and Demand at each node, plus Injection at Feeder
    3. Balances over the Day

    CC==True:
    All of the above plus
    4. Price for Flex at Feeder relative to regulation price
    5. SOC of all batteries
    6. Participation Factors over the day
    :return:
    """
    results_df = pandas.read_csv(path).iloc[:, 1:]
    results_df["Timestep"] = pandas.to_datetime(results_df["Timestep"])  # This is our x for all graphs
    ts_all = pandas.Series([i for i in range(num_time)])
    add = "CC " if CC else "Det "
    nodes = 16
    gens = [1, 2, 3, 4, 6, 7, 8, 11, 12, 13]

    # 1. Price for DA at Feeder relative to FT and GT
    price_dict = results_df[["FT", "Shadow Price 0", "GT"]].copy()
    price_dict.rename({'Shadow Price 0': 'Feeder DA Price'}, axis=1, inplace=True)
    output_file(filename=".\\plots\\" + add + "da_prices.html", title="Day-Ahead Prices")
    p = gen_plot(df=price_dict, ts=ts_all, title=add + 'Price for DA at Feeder relative to FT and GT',
                 x="Timestep t (hrs)", y="Prices (CHF/MWh)")
    show(p)

    # 1. Nodal Prices
    list_p = ["Shadow Price " + str(i) for i in range(nodes)]
    prices_dict = jitter_df(results_df[list_p], 0.0)
    output_file(filename=".\\plots\\" + add + "shadow_prices.html", title="Day-Ahead Shadow Prices")
    p = gen_plot(df=prices_dict, ts=ts_all, title=add + 'Shadow Prices across Nodes',
                 x="Timestep t (hrs)", y="Prices (CHF/kWh)")
    show(p)

    # 2. Supply and Demand at each node, plus Injection at Feeder
    results_df["Net Feeder"] = results_df["P Feed In"] - results_df["P Feed Out"]
    results_df["Net Renew"] = sum([results_df["DER" + str(i)] for i in gens])
    feeder_dict = jitter_df(results_df[["Net Feeder", "Net Renew", "P Local Gen", "P Local Dem"]], 0.00)
    output_file(filename=".\\plots\\" + add + "market.html", title="Supply and Demand")
    p = gen_plot(df=feeder_dict, ts=ts_all, title=add + 'Feeder Injection, Net Renewables, Local Gen and Demand',
                 x="Timestep t (hrs)", y="Power Flow (MW)")
    show(p)

    # 3. Balances over the Day
    list_b = ["Balance N" + str(i) for i in range(nodes)]
    balance_dict = jitter_df(results_df[list_b], 0.05)
    output_file(filename=".\\plots\\" + add + "balances.html", title="Balances over the Day")
    p = gen_plot(df=balance_dict, ts=ts_all, title=add + 'Balances over the Day',
                 x="Timestep t (hrs)", y="Account Balances (CHF)")
    show(p)

    # 4. SOC of all batteries
    list_soc = ["B" + str(i) for i in gens]
    soc_dict = jitter_df(results_df[list_soc], 0.01)
    output_file(filename=".\\plots\\" + add + "soc.html", title="SOCs")
    p = gen_plot(df=soc_dict, ts=ts_all, title=add + 'SOC of all Batteries',
                 x="Timestep t (hrs)", y="State of Charge (MWh)")
    show(p)

    if CC:
        if not NODAL:
            # 5. Price for Flex at Feeder relative to regulation price
            # Do vs Local Spot for now
            alpha_dict = results_df[["PI"]]
            output_file(filename=".\\plots\\" + "flexmarket.html", title="Alphas Price")
            p = gen_plot(df=alpha_dict, ts=ts_all, title=add + 'Alpha Price',
                         x="Timestep t (hrs)", y="Price (CHF/MWh)")
            show(p)

            # 6. Participation Factors over the day
            list_alpha = ["Alpha" + str(i) for i in range(nodes)]
            alphas_dict = jitter_df(results_df[list_alpha], 0.00)
            output_file(filename=".\\plots\\" + "alphas.html", title="Alphas Factors")
            p = gen_plot(df=alphas_dict, ts=ts_all, title=add + 'Participation Factors over the day',
                         x="Timestep t (hrs)", y="Price (CHF/MWh)")
            show(p)
        else:
            for ui in range(nodes):
                # 5. Price for Flex at Feeder relative to regulation price
                # Do vs Local Spot for now
                alpha_dict = results_df[["PI" + str(ui)]]
                output_file(filename=".\\plots\\" + "flexmarket.html", title="Alphas Price")
                p = gen_plot(df=alpha_dict, ts=ts_all, title=add + 'Alpha Price',
                             x="Timestep t (hrs)", y="Price (CHF/MWh)")
                show(p)

                # 6. Participation Factors over the day
                list_alpha = ["Alpha" + str(i) + " for " + str(ui) for i in range(nodes)]
                alphas_dict = jitter_df(results_df[list_alpha], 0.00)
                output_file(filename=".\\plots\\" + "alphas.html", title="Alphas Factors")
                p = gen_plot(df=alphas_dict, ts=ts_all, title=add + 'Participation Factors over the day',
                             x="Timestep t (hrs)", y="Price (CHF/MWh)")
                show(p)

            # 7. ALl flex prices at once
            list_flex = ["Shadow Price " + str(ui) for ui in range(nodes)]
            flexs_dict = jitter_df(results_df[list_flex], 0.0)
            output_file(filename=".\\plots\\" + "all_flexes.html", title="Alphas Prices")
            p = gen_plot(df=flexs_dict, ts=ts_all, title=add + 'Alpha Prices',
                         x="Timestep t (hrs)", y="Prices (CHF/MWh)")
            show(p)


def plot_results_red(path: str, num_time: int, CC: bool):
    """
    CC==False:
    1. Price for DA at Feeder relative to FT and GT
    2. Supply and Demand at each node, plus Injection at Feeder
    3. Balances over the Day

    CC==True:
    All of the above plus
    4. Price for Flex at Feeder relative to regulation price
    5. SOC of all batteries
    6. Participation Factors over the day
    :return:
    """
    results_df = pandas.read_csv(path).iloc[:, 1:]
    results_df["Timestep"] = pandas.to_datetime(results_df["Timestep"])  # This is our x for all graphs
    ts_all = pandas.Series([i for i in range(num_time)])
    add = "CC " if CC else "Det "
    nodes = 16
    gens = [1, 2, 3, 4, 6, 7, 8, 11, 12, 13]

    # 1. Price for DA at Feeder relative to FT and GT
    price_dict = results_df[["FT", "Shadow Price 0", "GT"]].copy()
    price_dict.rename({'Shadow Price 0': 'Feeder DA Price'}, axis=1, inplace=True)
    output_file(filename=".\\plots\\" + add + "da_prices.html", title="Day-Ahead Prices")
    p = gen_plot(df=price_dict, ts=ts_all, title=add + 'Price for DA at Feeder relative to FT and GT',
                 x="Timestep t (hrs)", y="Prices (CHF/MWh)")
    show(p)

    # 1. Nodal Prices
    list_p = ["Shadow Price " + str(i) for i in range(nodes)]
    prices_dict = jitter_df(results_df[list_p], 0.0)
    output_file(filename=".\\plots\\" + add + "shadow_prices.html", title="Day-Ahead Shadow Prices")
    p = gen_plot(df=prices_dict, ts=ts_all, title=add + 'Shadow Prices across Nodes',
                 x="Timestep t (hrs)", y="Prices (CHF/kWh)")
    show(p)

    # 2. Supply and Demand at each node, plus Injection at Feeder
    results_df["Net Feeder"] = results_df["P Feed"]
    results_df["Net Renew"] = sum([results_df["DER" + str(i)] for i in gens])
    feeder_dict = jitter_df(results_df[["Net Feeder", "Net Renew", "P Local Gen", "P Local Dem"]], 0.00)
    output_file(filename=".\\plots\\" + add + "market.html", title="Supply and Demand")
    p = gen_plot(df=feeder_dict, ts=ts_all, title=add + 'Feeder Injection, Net Renewables, Local Gen and Demand',
                 x="Timestep t (hrs)", y="Power Flow (MW)")
    show(p)

    # 3. Balances over the Day
    list_b = ["Balance N" + str(i) for i in range(nodes)]
    balance_dict = jitter_df(results_df[list_b], 0.05)
    output_file(filename=".\\plots\\" + add + "balances.html", title="Balances over the Day")
    p = gen_plot(df=balance_dict, ts=ts_all, title=add + 'Balances over the Day',
                 x="Timestep t (hrs)", y="Account Balances (CHF)")
    show(p)

    if CC:
        # 5. Price for Flex at Feeder relative to regulation price
        # Do vs Local Spot for now
        alpha_dict = results_df[["PI"]]
        output_file(filename=".\\plots\\" + "flexmarket.html", title="Alphas Price")
        p = gen_plot(df=alpha_dict, ts=ts_all, title=add + 'Alpha Price',
                     x="Timestep t (hrs)", y="Price (CHF/MWh)")
        show(p)

        # 6. Participation Factors over the day
        list_alpha = ["Alpha" + str(i) for i in range(nodes)]
        alphas_dict = jitter_df(results_df[list_alpha], 0.02)
        output_file(filename=".\\plots\\" + "alphas.html", title="Alphas Factors")
        p = gen_plot(df=alphas_dict, ts=ts_all, title=add + 'Participation Factors over the day',
                     x="Timestep t (hrs)", y="Price (CHF/MWh)")
        show(p)


class ConstraintFactory:
    """
    Make CVXPY Constraints
    """

    def __init__(self, i):
        self.cons = []
        self.id = {}
        self.user = i

    def create(self, con, name: str):
        """
        Add the constraint to the list, add its name to a dictionary to get the con_id
        :param con:
        :param name:
        """
        self.cons.append(con)
        self.id[name] = len(self.cons) - 1

    def getAll(self):
        """
        Get all the constraints
        :return:
        """
        return self.cons

    def getCon(self, name: str):
        """
        Get just one constraint with its name
        :param name:
        :return:
        """
        return self.cons[self.id[name]]


def dfs(visited: list, graph: list, ancestor: list, children: list, node: int):
    """
    This is the function used to travel through the tree
    Only go down the tree, so do not go to your ancestor
    :param ancestor: Update for your children
    :param children: Update for yourself
    :param visited: Tracks the order of visited nodes
    :param graph: List of all outflows for each node (yes, including to the ancestor)
    :param node: Current node
    :return:
    """
    if node not in visited:
        visited.append(node)
        for neigh in graph[node]:
            if neigh[1] != ancestor[node]:  # don't go back up
                children[node].append(neigh[1])
                ancestor[neigh[1]] = node  # only one per node anyway
                dfs(visited, graph, ancestor, children, neigh[1])


def get_ancestry(nodes: int, outflows: list, root: int):
    """
    We have a radial network with the root at node Root
    We want to establish the tree by determining for each node:
    1. Ancestor: What node is immediately upstream of you relative to the root (for the root, the answer is itself)
    2. Children: What nodes are immediately downstream of you relative to the root (for the leafs, the answer is no one)
    3. tree_order: which is the DFS queue for the tree
    :param nodes:
    :param outflows:
    :param root:
    :return:
    """
    # Need to determine the Ancestor and Children sets for all
    ancestor = [-1 for _ in range(nodes)]
    ancestor[root] = root
    visited = []
    children = [[] for _ in range(nodes)]

    # You need to start at the root
    # Via DFS
    dfs(visited, outflows, ancestor, children, root)

    return ancestor, children, visited


def jitter_df(df: pandas.DataFrame, std_ratio: float) -> pandas.DataFrame:
    """
    Add jitter to a DataFrame.

    Adds normal distributed jitter with mean 0 to each of the
    DataFrame's columns. The jitter's std is the column's std times
    `std_ratio`.

    From "https://stackoverflow.com/questions/40766909/suggestions-to-plot-overlapping-lines-in-matplotlib"
    by user "Florian Brucker"

    Returns the jittered DataFrame.
    """
    rng = default_rng()
    std = df.std().values * std_ratio
    jitter = pandas.DataFrame(
        std * rng.standard_normal(df.shape),
        index=df.index,
        columns=df.columns,
    )
    return df + jitter


def same_sign(a, b):
    """
    Find whehter two values have the same sign
    :param a:
    :param b:
    :return:
    """
    if (a < 0 and b < 0) or (a > 0 and b > 0):
        # if both are negative or positive, return true
        return True
    # if one is 0, then no penalty will occur ultimately (since either there is no total imbalance or you did nothing wrong)
    # so return false
    return False


def get_diff(a: float, b: float, error: float):
    """
    Determine the relative difference between a and b and whether said diff. is <= max allowed error
    ie.
    (a-b)/b <= error

    Return True/False and the caused absolute error
    """

    diff = abs(a - b)

    if diff <= error * abs(b):
        return True, diff

    return False, diff


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
