"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import math

import numpy as np


class Node:
    def __init__(self, action, obs, done, reward, state, mcts, parent=None):
        self.env = parent.env
        self.action = action  # Action used to go to this state

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.action_space_size = self.env.action_space.n
        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # Q
        self.child_priors = np.zeros([self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # N
        self.valid_actions = obs["action_mask"].astype(bool)

        self.reward = reward
        self.done = done
        self.state = state
        self.obs = obs

        self.mcts = mcts

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        # TODO (weak todo) add "softmax" version of the Q-value
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return (
            math.sqrt(self.number_visits)
            * self.child_priors
            / (1 + self.child_number_visits)
        )

    def best_action(self):
        """
        :return: action
        """
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        masked_child_score[~self.valid_actions] = -np.inf
        return np.argmax(masked_child_score)

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node.number_visits += 1
            current_node.total_value -= 1
            current_node = current_node.get_child(best_action)
        current_node.number_visits += 1
        current_node.total_value -= 1
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        if action not in self.children:
            self.env.set_state(self.state)
            obs, reward, done, _ = self.env.step(action)
            next_state = self.env.get_state()
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                obs=obs,
                mcts=self.mcts,
            )
        return self.children[action]

    def backup(self, value, multiplicity=1):
        current = self
        while current.parent is not None:
            current.total_value += value + multiplicity
            current = current.parent


class RootParentNode:
    def __init__(self, env):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.env = env


class MCTS:
    def __init__(self, model, mcts_param):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.max_batch_size = mcts_param["max_batch_size"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]

    def compute_action(self, node):

        num_sim = 0
        tuned_batch_size = min(self.max_batch_size, self.num_sims)
        while num_sim < self.num_sims:
            # Select a batch of leaves
            leaf_multiplicity = {}
            for _ in range(tuned_batch_size):
                leaf = node.select()
                if leaf in leaf_multiplicity:
                    leaf_multiplicity[leaf] += 1
                else:
                    leaf_multiplicity[leaf] = 1
            num_sim += tuned_batch_size
            batched_leaves = []
            for leaf in leaf_multiplicity:
                if leaf.done:
                    leaf.backup(leaf.reward, leaf_multiplicity[leaf])
                else:
                    batched_leaves.append(leaf)
            tuned_batch_size = min(int((len(batched_leaves)+1) * 1.2),
                self.max_batch_size, self.num_sims - num_sim)

            if len(batched_leaves) == 0:
                continue

            child_priors_batch, value_batch = self.model.compute_priors_and_value(
                [leaf.obs for leaf in batched_leaves])
            for leaf_idx, child_priors in enumerate(child_priors_batch):
                if self.add_dirichlet_noise:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet(
                        [self.dir_noise] * child_priors.size
                    )
                leaf = batched_leaves[leaf_idx]
                leaf.expand(child_priors)
                leaf.backup(value_batch[leaf_idx], leaf_multiplicity[leaf])

        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(
            tree_policy
        )  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = np.argmax(tree_policy)
        else:
            # otherwise sample an action according to tree policy probabilities
            action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.children[action]
