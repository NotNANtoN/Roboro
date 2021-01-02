import math
import numpy as np
import torch

class Node():

    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = -1
        self.children = {}
        self.reward = 0
        self.hidden_obs = None
        self.prior = prior

    @property
    def expanded(self):
        return len(self.children)

    def value(self):
        if self.visit_count:
            return self.value_sum / self.visit_count
        return 0

    def expand(self, actions, reward, policy_q_val, hidden_obs):
        self.reward = reward
        self.hidden_obs = hidden_obs
        policy  = {a: policy_q_val.flatten()[i] for i, a in enumerate(actions)}
        for action, _p in policy.items():
            self.children[action] = Node(_p)

class MCTS():

    def __init__(self, num_simulations):
        self.num_simulations = num_simulations
        self.discount = 0.98
          # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


    def run(self, model, obs):
        root = Node(0)
        model.eval()
        obs = model.normalizer.norm(obs)
        root_obs_h = model.representation_model(obs)
        root_q_val, root_value = model.prediction_model(root_obs_h)
        init_a = torch.argmax(root_q_val, dim=1).unsqueeze(1)
        obs_h_next_pred, pred_reward =  model.dynamics_model(root_obs_h, init_a)

        root.expand([0,1], pred_reward, root_q_val, root_obs_h)

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded:
                current_tree_depth += 1
                action, node  = self.select_child(node, min_max_stats)
                search_path.append(node)

            parent = search_path[-2]
            obs_h, pred_reward =  model.dynamics_model(parent.hidden_obs, torch.tensor([[action]]).to(parent.hidden_obs.device))
            policy_q_val, value = model.prediction_model(obs_h)

            node.expand([0, 1], pred_reward, policy_q_val, obs_h)
            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_value,
        }

        return root, extra_info

    def select_child(self, node, min_max_stats):
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = np.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def backpropagate(self, search_path, value, min_max_stats):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.discount * node.value())

            value = node.reward + self.discount * value
    def ucb_score(self, parent, child, min_max_stats):
        pb_c = (
            math.log(
                (parent.visit_count + self.pb_c_base + 1) / self.pb_c_base
            )
            + self.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.discount
                * child.value()
            )
        else:
            value_score = 0

        return prior_score + value_score



class MinMaxStats:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
