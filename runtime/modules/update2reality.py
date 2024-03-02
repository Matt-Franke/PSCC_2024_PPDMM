"""
This supports the work on developing a way to update injections and flows to the response policies and realized errors
"""
from modules import inputs, trivia


def gen_order(anc, kids, n):
    queue = [i for i in range(n) if not kids[i]]  # init to all the leaves!
    visited = []

    while queue:
        curr = queue.pop(0)
        check = sum([0 if j in visited else 1 for j in kids[curr]])
        if curr not in visited and not check:
            # to repeat double visits, which should not happen anyways
            one_up = curr

            while not check:
                # keep going up until our ancestor has unresolved children
                # this will resolve previously unresolved chains and that is fine
                visited.append(one_up)
                a = anc[one_up]

                if a == one_up:
                    # we have reached the feeder
                    check = 0
                    break

                check = sum([0 if j in visited else 1 for j in kids[a]])
                one_up = a

            if check:
                # at least one child has not been visited
                # thus we need to wait to process the final ancestor in this chain
                # note that we keep updating one_up so it will be that final ancestor
                queue.append(one_up)

    return visited


if __name__ == "__main__":
    joint_df, tariff = inputs.retrieve_basic_inputs()
    incidence, resistance, reactance, susceptance, limits, nodes, lines, pf, der_cost = inputs.get_basic_network()
    connect, outflows = inputs.generate_connect_outflows(nodes, lines, incidence)
    ancestor, children, tree_order = trivia.get_ancestry(nodes, outflows, 0)

    order = gen_order(ancestor, children, nodes)
    print(order)
