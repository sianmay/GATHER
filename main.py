import argparse
import sys
from src.Gather import Gather, EnvResourceComponent
from src.Agents import *
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import multiprocessing
from itertools import product

COUNT = 0
SEED = 5

ENV_SIZE = 50
NEST_SIZE = 4
NUM_AGENTS = 20

DEPOSIT_RATE = 0.25
FDECAY_RATE = 0.04
HDECAY_RATE = 0.00

ITERATIONS = 1001

COST = 0
COST_FREQUENCY = 100000

VIS = False

ENV_MODE = 0
NETWORK_MODE = 0
MOVE_MODE = 'PMS'

GAMMA = 0.99
ETA = 0.1
RANDOM_CHANCE = 0.0

y_list = []
sn_list = []
graph_list = []

make_friend = 0
lose_friend = 0


def has_alliance(i, A):
    for j in range(NUM_AGENTS):
        if i != j:
            if A[i, j] > 0 and A[j, i] > 0:
                return True
    return False


def follower_number(j, A):
    fave = np.argmax(A, axis=1)
    followers = 0
    for i in range(NUM_AGENTS):
        if i != j:
            if (fave[i] == j or A[i, j] > 0) and A[j, i] < 0:
                followers += 1
    return followers


def get_friend_amnt(i, A):
    friends = 0
    for j in range(NUM_AGENTS):
        if i != j:
            if A[i, j] > 0 and A[j, i] > 0:
                friends += 1
    return friends


def plot_self_op_friends(self_op_list, friends_list):
    fig, axs = plt.subplots(2)
    axs[0].plot(np.array(self_op_list))
    axs[0].set_ylabel("Agent's self opinion")
    axs[1].plot(np.array(friends_list))
    axs[1].set_xlabel("Timestep (t)")
    axs[1].set_ylabel("Agent's #allies")
    axs[0].set_ylim([-1.1, 1.1])

    plt.show()


def draw_graph(G, ax):
    # Compute the in-degree centrality for each node based on weighted edges
    degree = [G.in_degree(n, weight="weight") for n in G.nodes()]

    edge_nodes = set(G)  # - {center_node}
    # Ensures the nodes around the circle are evenly distributed
    pos = nx.circular_layout(G.subgraph(edge_nodes))

    node_weights = nx.get_node_attributes(G, 'weight')
    i = 0

    for node, weight in node_weights.items():
        radius = 1
        if degree[i] != 0 and degree[i] != sum(degree):
            radius = np.maximum(1 - ((degree[i] * 3) / sum(degree)), 0.1)
        x, y = pos[node]
        x = x * radius
        y = y * radius
        pos[node] = (x, y)
        i += 1

    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Draw the graph with colored edges based on weights
    pos_edges = [(u, v) for (u, v, w) in G.edges(data='weight') if w >= 0]
    colors = [G.nodes[n]['weight'] for n in G.nodes]

    plt.sca(ax)
    nodes = nx.draw_networkx_nodes(G, node_color=colors, cmap='OrRd', vmin=0, vmax=1, pos=pos, ax=ax, node_size=10,
                                   linewidths=0.25)
    nodes.set_edgecolor('black')
    nx.draw_networkx_edges(G, pos=pos, edgelist=pos_edges, edge_color=[edge_weights[e] for e in pos_edges],
                           edge_cmap=plt.cm.OrRd, edge_vmin=0, edge_vmax=1, ax=ax, node_size=10, arrowsize=5)


def makeNetwork(matrix):
    matrix += 1
    matrix /= 2
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if i != j and matrix[i, j] < 0.3:
                matrix[i][j] = 0
    G = nx.DiGraph(np.array(matrix))
    for i in range(G.number_of_nodes()):
        G.nodes[i]['weight'] = matrix[i, i]
        if matrix[i, i] != 0:
            G.remove_edge(i, i)
    return G


def get_social_network(n):
    return np.zeros((n, n))


def get_communication_network(mode: int, num_agents):
    if mode == 0:
        print('MODE: Zero Communication')
        return np.zeros((num_agents, num_agents))
    elif mode == 1:
        print('MODE: Full Communication')
        return np.ones((num_agents, num_agents))
    elif mode == 2:
        print('MODE: Self Communication')
        net = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            net[i][i] = 1.0
        return net
    elif mode == 3:
        print('MODE: Ring Communication')
        net = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            net[i][i] = 1.0
            net[i][i - 1] = 1.0
            net[i][(i + 1) % num_agents] = 1.0
        return net
    elif mode == 4:
        print('MODE: Authoritarian Communication')
        net = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            if i < 0.1 * num_agents:
                net[i] = 1.0
            else:
                net[i][i] = 1.0
        return net
    elif mode == 5:
        print('MODE: Other Communication')
        net = np.ones((num_agents, num_agents))
        for i in range(num_agents):
            net[i][i] = 0.0
        return net
    elif mode == 6:
        print('MODE: Islands Communication')
        net = np.zeros((num_agents, num_agents))
        modulo = int(num_agents // 5)
        for i in range(num_agents):
            for j in range(num_agents):
                if i % modulo == j % modulo:
                    net[i][j] = 1.0
        return net
    return None


def gini(data):
    sorted_x = np.sort(data)
    n = len(sorted_x)
    cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0.0 else 0.0


def parseArgs():
    """Create GATHER Parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size',
                        help='The size of the environment. In all instances the environment will be a square gridworld.',
                        default=ENV_SIZE, type=int)
    parser.add_argument('-n', '--nest', help='The size of the nest. In all instances the nest will be square.',
                        default=NEST_SIZE, type=int)
    parser.add_argument('-a', '--agents', help='The number of agents to initialize.', default=NUM_AGENTS, type=int)
    parser.add_argument('--deposit', help='The amount of pheromone dropped by the agents.', default=DEPOSIT_RATE,
                        type=float)
    parser.add_argument('--hdecay', help='The rate at which the home pheromone evaporates.', default=HDECAY_RATE,
                        type=float)
    parser.add_argument('--fdecay', help='The rate at which the food pheromone evaporates.', default=FDECAY_RATE,
                        type=float)
    parser.add_argument('-i', '--iterations', help='How long the simulation should run for.',
                        default=ITERATIONS, type=int)
    parser.add_argument('--mode', help='What mode the environment should be initialized to', default=ENV_MODE, type=int)
    parser.add_argument('--network', help='The type of network to initialize', default=NETWORK_MODE, type=int)
    parser.add_argument('-v', '--visualize',
                        help='Whether environment images should be written to the output/env/ directory',
                        action='store_true')
    parser.add_argument('--seed', help="Specify the seed for the Model's pseudorandom number generator", default=None,
                        type=int)
    parser.add_argument('--move', help='Which agent movement system to use', default=MOVE_MODE, type=str)
    parser.add_argument('--gamma', help='Discount Factor for RL Systems', default=GAMMA, type=float)
    parser.add_argument('--eta', help='Learning Rate for RL Systems', default=ETA, type=float)
    parser.add_argument('--random', help='Chance for agent to take a random action.', default=RANDOM_CHANCE,
                        type=float)
    parser.add_argument('--detect', help='Should the agents detect and avoid being crowded with other agents',
                        action='store_true')
    parser.add_argument('--center', help='Should the home base be centered in the environment.',
                        action='store_true')
    parser.add_argument('--tpa', help='Set strength of Task Performance Appreciation', default=0.4,
                        type=float)
    parser.add_argument('--rho', help='Set strength of Opinion Propagation', default=0.9,
                        type=float)
    parser.add_argument('--omega', help='Set strength of Opinion Reciprocation', default=0.6,
                        type=float)
    parser.add_argument('--sigma', help='Set Opinion Propagation Coefficient value (for sigmoid function)', default=0.3,
                        type=float)
    parser.add_argument('--delta', help='Set randomness strength in opinion dynamics', default=0.1,
                        type=float)
    parser.add_argument('--k', help='Set number of agents spoken about within an interaction', default=5,
                        type=int)
    parser.add_argument('--save_network',
                        help='Whether emergent network should be written to the network_data/ directory',
                        action='store_true')
    parser.add_argument('--visualize_network',
                        help='Whether emergent network should be written to the output/networks/ directory',
                        action='store_true')
    parser.add_argument('--step', help='Set the iteration step size between visualizations', default=1,
                        type=int)

    return parser.parse_args()


def main(params=None):
    grid = False

    args = parseArgs()
    communication_network = get_communication_network(args.network, args.agents)
    social_network = get_social_network(args.agents)

    if not grid:
        setTPA(args.tpa)
        setRho(args.rho)
        setOmega(args.omega)
        setSigma(args.sigma)
        setDelta(args.delta)
        setK(args.k)

    # Create Model
    model = Gather(args.size, args.nest, args.deposit, args.hdecay, args.fdecay, communication_network, social_network,
                   COST, COST_FREQUENCY, environment_mode=args.mode, seed=args.seed, move_mode=args.move,
                   gamma=args.gamma, eta=args.eta, random_chance=args.random, detect=args.detect, center=args.center)

    # Add Agents to the environment
    for i in range(args.agents):
        model.environment.add_agent(AntAgent(i, model), *model.random.choice(model.home_locs))

    # Run Model
    agent_val = len(model.resource_distribution) + 1
    for i in range(args.iterations):
        model.execute()
        wealth_arr = np.array([agent[ResourceComponent].wealth for agent in model.environment])
        collected = model.environment[EnvResourceComponent].resources  # - collected

        # if i > 0 and i % 100 == 0:
        # plot_self_op_friends(self_op_i, friends_i)
        # save_matrix("#followers", followers_cnt)
        #   leader_op_plot4(friends_i, leader_rep, ally_rep, av_rep)

        print(f'\r{i + 1}/{args.iterations} - Collected: {collected} Gini: {gini(wealth_arr)}',
              file=sys.stdout, flush=True, end='\r')

        if args.visualize_network and i % args.step == 0 and i > 0:
            mod, cenw, cen = centrality_modularity(social_network.copy())
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            G = makeNetwork(social_network.copy())
            draw_graph(G, ax1)
            # plt.gcf()
            norm = colors.Normalize(vmin=-1, vmax=1)
            # ax2.matshow(social_network, cmap='seismic', norm=norm)
            ax2.matshow(social_network, cmap='OrRd', norm=norm)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='OrRd'), ax=ax2, aspect=30)
            # pattern = patterns.find_pattern(social_network)
            mod = format(mod, '.2f')
            cenw = format(cenw, '.2f')
            fig.suptitle(f'Iteration {i}\nModularity={mod}, Centralisation={cenw}')  # , {pattern}')
            fig.savefig(f'./output/networks/iteration_{i}.png')
            plt.close()

        if args.visualize and (i % args.step == 0):
            # Will generate a series of figures of the environment.
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)
            img = np.copy(model.environment.cells[Gather.RESOURCE_KEY].to_numpy()).reshape(ENV_SIZE, ENV_SIZE)

            for agent in model.environment:
                x, y = agent[ENV.PositionComponent].xy()
                img[y][x] = agent_val

            ax1.imshow(img, cmap='Set1')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            fcells = np.zeros(model.environment.width ** 2)
            hcells = np.zeros(model.environment.width ** 2)

            for agent in model.environment:
                fcells += agent[PheromoneComponent].f_pheromones
                hcells += agent[PheromoneComponent].h_pheromones

            img = fcells.reshape(ENV_SIZE, ENV_SIZE)
            ax2.imshow(img)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')

            img = hcells.reshape(ENV_SIZE, ENV_SIZE)
            ax3.imshow(img)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')

            fig.suptitle(f'Iteration {i}')
            fig.savefig(f'./out_env11_cluster/iteration_{i}.png')
            plt.close(fig)

    if args.save_network:
        result = model.environment[EnvResourceComponent].resources
        name = f"seed={args.seed}, iter={i}, rho={args.rho}, tpa={args.tpa}, omega={args.omega}"
        # name = f"i={i}, Result: {result}, Params: {params}"
        save_network(name, social_network)

    mod, cenw, cen = centrality_modularity(social_network.copy())
    print()
    print("Emergent Network Modularity: ", mod)
    print("Emergent Network Centralisation: ", cenw)
    return model.environment[EnvResourceComponent].resources, None, social_network, args.seed


def centrality(G):
    N = G.order()
    indegrees = [G.in_degree(n, weight="weight") for n in G.nodes()]  # Get in-degrees with weighted edges
    max_in = np.max(indegrees)
    cenw = float((N * max_in - sum(indegrees))) / ((N - 1) ** 2)
    indegrees = [G.in_degree(n) for n in G.nodes()]  # Get in-degrees with weighted edges
    max_in = np.max(indegrees)
    cen = float((N * max_in - sum(indegrees))) / ((N - 1) ** 2)
    return cenw, cen


def centrality_modularity(matrix):
    for i in range(0, len(matrix[0])):
        for j in range(0, len(matrix[0])):
            matrix[i, j] += 1
            matrix[i, j] /= 2
    G = nx.DiGraph(np.array(matrix))
    for i in range(G.number_of_nodes()):
        G.nodes[i]['weight'] = matrix[i, i]
    degree2 = [G.in_degree(n) for n in G.nodes()]
    max2 = np.max(degree2)
    modularity = 1
    if max2 != 0:
        com = nx.community.louvain_communities(G, resolution=0, seed=123)
        modularity = nx.community.modularity(G, com)
    cenw, cen = centrality(G)
    return modularity, cenw, cen


def change_to_zero(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 1:
                matrix[i, j] = 0
    return matrix


def save_network(name, matrix):
    # Define the filename and path
    filename = name + ".txt"
    # path = "./networks/noneg/"
    path = "./network_data/"
    # Write the matrix to a file
    np.savetxt(path + filename, matrix, fmt="%f")
    print("Matrix has been written to", path + filename)


def process_params(params):
    global SEED, NUM_AGENTS, COUNT
    seed, TPA, opinion_inf, vanity, sigma, delta, k = params
    SEED = seed
    setTPA(TPA)
    setRho(opinion_inf)
    setOmega(vanity)
    setSigma(sigma)
    setDelta(delta)
    setK(k)
    result, y, network, seed = main(params)
    thread_id = multiprocessing.current_process().name
    print('params: ', params)
    return result, thread_id, y, params, network


def para_grid():
    args = parseArgs()
    # Define the values for each parameter
    TPA_values = [0.4]  # [0.0, 0.2, 0.4, 0.6, 0.8, 1]
    opinion_inf_values = [0.9]  # [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]#[0.25, 0.5, 0.75, 1.0]
    vanity_values = [0.6]  # [0.2, 0.4, 0.6, 0.8] #[0.0, 0.5, 1.0]
    sigma_values = [0.3]  # , 0.5]
    delta_values = [0.1]
    k_values = [5]  # , 5, 10]#, 10]
    # Create a list of all possible combinations of the parameter values
    parameter_combinations = product([args.seed], TPA_values, opinion_inf_values, vanity_values, sigma_values,
                                     delta_values, k_values)
    # Create a pool of workers
    pool = multiprocessing.Pool()
    # Apply the process_params function to each combination of parameters using the pool of workers
    results = pool.map(process_params, parameter_combinations)
    # Close the pool of workers and wait for them to finish
    pool.close()
    pool.join()
    for result, thread_id, y, params, network in results:
        print(f"Thread ID: {thread_id}, Result: {result}, Params: {params}")
    return list


if __name__ == '__main__':
    main()
