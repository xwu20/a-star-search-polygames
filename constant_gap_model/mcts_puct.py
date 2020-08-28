import numpy as np
import scipy.stats
import copy
import hydra
import pickle

import logging
log = logging.getLogger(__file__)

from utils import Logger

from multiprocessing import Pool, Process, Queue

import os
import sys


from random import randrange
from scipy.special import softmax


class Node:
	#states are [depth, yes/no optimal path, value network value, current average, number of times visited]
    def __init__(self, node_id, depth, opt_bool, vn_value, q_value, visit_count):
        self.node_id = node_id
        self.depth = depth
        self.opt_bool = opt_bool
        self.vn_value = vn_value
        self.q_value = q_value
        self.visit_count = visit_count
        self.children = []

    def add_child(self, node_id):
        self.children.append(node_id)



def initializeNoiseRVs(args):
    noise_rvs = []

    for i in range(args.depth):
        if args.noise_type == "exp":
            sd = float(1)/float(np.power(args.alpha, i+1))
        elif args.noise_type == "poly":
            sd = float(1) / float(np.power(i+1, args.alpha))
        else:
            raise NotImplementedError(f"Unknown noise_type: {args.noise_type}")

        rv = scipy.stats.norm(loc=0, scale=sd)
        noise_rvs.append(rv)

    rv = scipy.stats.norm(loc=0, scale=0)
    noise_rvs.append(rv)

    return noise_rvs



def mcts_benchmark(noise_rvs, seed, args):
	node_dict = {}
	expand_count = 0

	total_data_points = int(args.max_expansions / args.test_interval)
	chosen_action_grid = np.zeros((total_data_points,))


	root_id = 0
	root_depth = 0
	root_opt_bool = 1
	noise_seed = np.random.seed([0, seed, 0])
	noise_val = noise_rvs[0].rvs()
	root_vn_value = args.opt + noise_val
	print(root_vn_value)
	root_visit_count = 1
	root = Node(root_id, root_depth, root_opt_bool, root_vn_value, root_vn_value, root_visit_count)

	node_dict[root_id] = root


	while expand_count < args.max_expansions:
		current_node = node_dict[0]
		expand_current_node = 0

		path_states = [0]

		while expand_current_node == 0:

			softmax_prob = np.zeros(args.degree)

			child_depth = current_node.depth + 1

			for d in range(args.degree):
				child_id = current_node.node_id * args.degree + d + 1
				policy_noise_seed = np.random.seed([child_id, seed, 1])
				policy_noise_val = noise_rvs[child_depth].rvs()

				if current_node.opt_bool == 1 and d == 0:
					PN_val = args.opt + policy_noise_val
				else:
					PN_val = args.opt - args.eta + policy_noise_val

				softmax_prob[d] = args.temperature*PN_val
			softmax_prob = softmax(softmax_prob)


			max_ucb = np.NINF
			child_node = None
			for d in range(args.degree):
				child_id = current_node.node_id * args.degree + d + 1
				if child_id in node_dict:
					child_state_node = node_dict[child_id]
					child_ucb = child_state_node.q_value + args.constant*softmax_prob[d]*np.sqrt(current_node.visit_count)*(float(1)/float(1+child_state_node.visit_count))
				else:
					child_ucb = current_node.vn_value + args.constant*softmax_prob[d]*np.sqrt(current_node.visit_count)

				if child_ucb > max_ucb:
					max_ucb = child_ucb
					child_node = child_id

			if child_node in node_dict:
				path_states.append(child_node)
				current_node = node_dict[child_node]
			else:
				expand_current_node = 1
				expand_count += 1

				if current_node.depth < args.depth - 1:
					new_id = child_node
					new_depth = current_node.depth + 1
					new_visit_count = 1
					new_opt_bool = 0
					new_vn_value = args.opt - args.eta

					if (current_node.opt_bool == 1) and (child_node == current_node.node_id*args.degree + 1):
						new_vn_value = args.opt
						new_opt_bool = 1

					noise_seed = np.random.seed([new_id, seed, 0])
					noise_val = noise_rvs[new_depth].rvs()
					new_vn_value += noise_val

					new_child = Node(new_id, new_depth, new_opt_bool, new_vn_value, new_vn_value, new_visit_count)
					node_dict[new_id] = new_child
					node_dict[current_node.node_id].children.append(new_id)
				else:
					new_vn_value = current_node.vn_value

		for id_val in path_states:
			state_node = node_dict[id_val]


			new_q_val = (state_node.q_value*state_node.visit_count + new_vn_value) / float(state_node.visit_count + 1)
			# print(" neq q value is:")
			# print(new_q_val)

			node_dict[id_val].q_value = new_q_val
			node_dict[id_val].visit_count += 1

		if (expand_count%args.test_interval == 0):
			print("expand count is")
			print(expand_count)
			max_count_index = -1
			max_count_number = np.NINF
			number_probed_actions = len(node_dict[0].children)
			for c in range(number_probed_actions):
				child_index = node_dict[0].children[c]
				child_node = node_dict[child_index]
				if child_node.visit_count > max_count_number:
					max_count_number = child_node.visit_count
					max_count_index = child_index
			print(max_count_index)

			if max_count_index == 1:
				chosen_action_grid[int(expand_count / args.test_interval)-1] = 1
			else:
				chosen_action_grid[int(expand_count / args.test_interval)-1] = 0

	return chosen_action_grid

class MyProcess(Process):
    def __init__(self, noise_rvs, tri, args, q):
        super().__init__()
        self.noise_rvs = noise_rvs
        self.tri = tri
        self.args = args
        self.q = q

    def run(self):
        sys.stdout = Logger(f"./{self.tri}.log", write_to_terminal=False)
        sys.stderr = Logger(f"./{self.tri}.err", write_to_terminal=False)

        args = self.args
        this_chosen_action_grid = mcts_benchmark(self.noise_rvs, self.tri, self.args)

        self.q.put((self.tri, this_chosen_action_grid)) 

@hydra.main(config_path='conf/first_level.yaml', strict=True)
def main(args):
    cmd_line = " ".join(sys.argv)
    log.info(f"{cmd_line}")
    log.info(f"Working dir: {os.getcwd()}")


    total_data_points = int(args.max_expansions / args.test_interval)
    chosen_action_grid = np.zeros((args.num_trials, total_data_points))

    noise_rvs = initializeNoiseRVs(args)

    q = Queue()
    processes = []
    for tri in range(args.num_trials):
        p = MyProcess(noise_rvs, tri, args, q)
        p.start()
        processes.append(p)

    for i in range(args.num_trials):
        tri, this_chosen_action_grid = q.get()
        chosen_action_grid[tri,:] = this_chosen_action_grid
        log.info(f"#finished: {i+1}/{args.num_trials}")

    for p in processes:
        p.join()

    log.info(chosen_action_grid)

    mean_chosen_action = np.mean(chosen_action_grid, axis=0)
    std_chosen_action = np.std(chosen_action_grid, axis=0)

    log.info("mcts with value and policy network eta model")

    pickle.dump(dict(ca_grid=chosen_action_grid, mean_ca=mean_chosen_action, std_ca=std_chosen_action), open("stats.pkl", "wb"))

    log.info(mean_chosen_action)
    log.info(std_chosen_action)

    log.info(f"Working dir: {os.getcwd()}")

if __name__ == "__main__":
    main()

# root_state_node = node_dict[0]
# num_children = len(root_state_node.children)
# print("number of children")
# print(num_children)

# for c in range(num_children):
# 	child_index = root_state_node.children[c]
# 	child_node = node_dict[child_index]
# 	print("is this child optimal")
# 	print(child_node.opt_bool)
# 	print(child_node.node_id)
# 	print(child_node.vn_value)
# 	print("how many times was this child picked")
# 	print(child_node.visit_count)





