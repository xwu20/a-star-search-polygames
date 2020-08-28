import numpy as np
import scipy.stats
import copy
import hydra
import pickle
import queue

import logging
log = logging.getLogger(__file__)

from utils import Logger

from multiprocessing import Pool, Process, Queue

import os
import sys

def initializeNoiseRVsAndCdBonus(args):
    noise_rvs = []
    c_d_bonus = []

    for i in range(args.depth):
        if args.noise_type == "exp":
            sd = float(1)/float(np.power(args.alpha, i+1))
        elif args.noise_type == "poly":
            sd = float(1) / float(np.power(i+1, args.alpha))
        else:
            raise NotImplementedError(f"Unknown noise_type: {args.noise_type}")

        rv = scipy.stats.norm(loc=0, scale=sd)
        noise_rvs.append(rv)
        c_d_bonus.append(args.cd_constant*np.sqrt(i+1)*sd)

    rv = scipy.stats.norm(loc=0, scale=0)
    noise_rvs.append(rv)
    c_d_bonus.append(0.0)

    return noise_rvs, c_d_bonus


def exactVNAlg(noise_rvs, c_d_bonus, seed, args):
        count = 0
        total_data_points = int(args.max_expansions / args.test_interval)
        chosen_action_grid = np.zeros((total_data_points,))

        L = queue.PriorityQueue()
        #states are [-1*UCB value (0), depth (1), yes/no optimal path (2), value network value (3), opt_one_step (4), node_id (5)]
        noise_seed = np.random.seed([0, seed, 0])
        noise_val = noise_rvs[0].rvs()
        s_0_val = args.opt + noise_val
        s_0 = (-1*(s_0_val + c_d_bonus[0]), 0, 1, s_0_val, -1, 0)
        count += 1
        L.put(s_0)
        MIN = s_0_val - c_d_bonus[0]

        first_elt = L.get()

        first_elt_depth = first_elt[1]
        while first_elt_depth < args.depth and count < args.max_expansions:

                child_depth = first_elt_depth + 1

                policy_values = []

                for n in range(args.degree):
                        nod_id = first_elt[5]*args.degree + n + 1
                        policy_noise_seed = np.random.seed([nod_id, seed, 1])
                        policy_noise_val = noise_rvs[child_depth].rvs()

                        if first_elt[2] == 1 and n == 0:
                                PN_val = args.opt + policy_noise_val
                        else:
                                PN_val = args.opt - args.eta + policy_noise_val

                        policy_values.append([nod_id, PN_val])

                policy_values = sorted(policy_values, key=lambda tup: tup[1], reverse=True)


                for n in range(args.degree):
                        nod_id = policy_values[n][0]
                        value_noise_seed = np.random.seed([nod_id, seed, 0])
                        value_noise_val = noise_rvs[child_depth].rvs()
                        if n < 2 or (policy_values[0][1] - policy_values[n-1][1] < 2*c_d_bonus[child_depth]):
                                if (first_elt[2] == 1) and (nod_id == first_elt[5]*args.degree + 1):
                                        VN_val = args.opt + value_noise_val
                                        state = (-1*(VN_val + c_d_bonus[child_depth]), child_depth, 1, VN_val)
                                else:
                                        VN_val = args.opt - args.eta + value_noise_val
                                        state = (-1*(VN_val + c_d_bonus[child_depth]), child_depth, 0, VN_val)

                                count += 1
                                if child_depth == 1:
                                        drawn_index = int(nod_id - first_elt[5]*args.degree - 1)
                                        state += (drawn_index + 1,)
                                else:
                                        state += (first_elt[4],)

                                state += (nod_id,)
                                L.put(state)

                                if (count%args.test_interval == 0):
                                    L_new = queue.PriorityQueue()
                                    best_so_far_value = np.NINF
                                    best_so_far_state = None
                                    while not L.empty():
                                        new_elt = L.get()
                                        if -1*new_elt[0] >= MIN:
                                            L_new.put(new_elt)
                                        vn_value_constant = new_elt[3]
                                        if vn_value_constant > best_so_far_value:
                                            best_so_far_value = vn_value_constant
                                            best_so_far_state = new_elt
                                    L = L_new
                                    print(count)
                                    print(best_so_far_state[4])
                                    if best_so_far_state[4] == 1:
                                        chosen_action_grid[int(count / args.test_interval)-1] = 1
                                    else:
                                        chosen_action_grid[int(count / args.test_interval)-1] = 0

                                if VN_val - c_d_bonus[child_depth] > MIN:
                                    MIN = VN_val - c_d_bonus[child_depth]
                        else:
                            break
                if not L.empty():
                    first_elt = L.get()
                    first_elt_depth = first_elt[1]
                else:
                    break

        return first_elt, chosen_action_grid, count

class MyProcess(Process):
    def __init__(self, noise_rvs, c_d_bonus, tri, args, q):
        super().__init__()
        self.noise_rvs = noise_rvs
        self.c_d_bonus = c_d_bonus
        self.tri = tri
        self.args = args
        self.q = q

    def run(self):
        sys.stdout = Logger(f"./{self.tri}.log", write_to_terminal=False)
        sys.stderr = Logger(f"./{self.tri}.err", write_to_terminal=False)

        args = self.args

        _, this_chosen_action_grid, count = exactVNAlg(self.noise_rvs, self.c_d_bonus, self.tri, self.args)
        total_data_points = int(args.max_expansions / args.test_interval)

        # print(count)
        if count < args.max_expansions:
                last_index = int(count / args.test_interval)

                for fil in range(last_index, total_data_points):
                        this_chosen_action_grid[fil] = this_chosen_action_grid[last_index-1]

        self.q.put((self.tri, this_chosen_action_grid, count)) 

@hydra.main(config_path='conf/new_with_policy.yaml', strict=True)
def main(args):
    cmd_line = " ".join(sys.argv)
    log.info(f"{cmd_line}")
    log.info(f"Working dir: {os.getcwd()}")

    #num_leaves = np.power(degree, depth)

    #rewards = [-1]*num_leaves
    total_data_points = int(args.max_expansions / args.test_interval)
    chosen_action_grid = np.zeros((args.num_trials, total_data_points))
    counts_data = np.zeros(args.num_trials)
    # def initializeRewards():
    # 	global rewards
    # 	for i in range(num_leaves):
    # 		rewards[i] = opt - eta
    # 	rewards[0] = eta

    noise_rvs, c_d_bonus = initializeNoiseRVsAndCdBonus(args)

    q = Queue()
    processes = []
    for tri in range(args.num_trials):
        p = MyProcess(noise_rvs, c_d_bonus, tri, args, q)
        p.start()
        processes.append(p)

    for i in range(args.num_trials):
        tri, this_chosen_action_grid, count = q.get()
        counts_data[tri] = count
        chosen_action_grid[tri,:] = this_chosen_action_grid
        log.info(f"#finished: {i+1}/{args.num_trials}")

    for p in processes:
        p.join()

    log.info(chosen_action_grid)
    log.info(counts_data)

    mean_chosen_action = np.mean(chosen_action_grid, axis=0)
    std_chosen_action = np.std(chosen_action_grid, axis=0)

    mean_counts = np.mean(counts_data)

    log.info("new algorithm with policy eta model")

    pickle.dump(dict(ca_grid=chosen_action_grid, mean_ca=mean_chosen_action, std_ca=std_chosen_action, mean_counts=mean_counts), open("stats.pkl", "wb"))

    log.info(mean_chosen_action)
    log.info(std_chosen_action)
    log.info(mean_counts)

    log.info(f"Working dir: {os.getcwd()}")

if __name__ == "__main__":
    main()
