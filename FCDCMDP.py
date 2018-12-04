from copy import deepcopy
from typing import Dict, Any, List
import itertools

from NP import PropagationTree
from NPExtension import TwitterNetworkPropagation

from mdptoolbox.mdp import FiniteHorizon
import numpy as np
from scipy.misc import comb
from Baselines import *


def b2i(binary_tuple):
    l = len(binary_tuple)
    return np.sum(np.asarray(binary_tuple) * np.asarray([2 ** (l - i - 1) for i in range(l)]))


def i2b(integer, max_len):
    b = [int(x) for x in list("{0:#b}".format(integer)[2:])]
    return np.asarray([0 for _ in range(max_len - len(b))] + b)


class FCDCMDP(WeightedUserModel):

    def __init__(self, num_news, budget, mdp_cls=None, is_verbose=False, **mdp_kwargs):
        super().__init__()
        self.mdp = None
        self.mdp_cls = mdp_cls if mdp_cls else FiniteHorizon
        self.mdp_kwargs = mdp_kwargs

        self.num_news = num_news
        self.budget = budget
        self.is_verbose = is_verbose

        # (a, a)
        self.actions = self.get_actions()

        # (a, s, s)
        self.transition = self.get_transition()

        self.reward = None

    def select_fake_news(self,
                         active_news_to_tree: Dict[Any, PropagationTree],
                         network_propagation: TwitterNetworkPropagation,
                         budget: int,
                         select_exact: bool) -> List[Any]:

        if self.mdp_kwargs["N"] is None:
            self.mdp_kwargs["N"] = 1

        self.reward = self.get_reward(active_news_to_tree, network_propagation)
        self.mdp = self.mdp_cls(self.transition, self.reward, **self.mdp_kwargs)
        if self.is_verbose:
            self.mdp.setVerbose()
        self.mdp.run()

        current_state = b2i(np.asarray([0 if is_not_checked_and_not_blocked(network_propagation, info) else 1
                                        for info in range(len(active_news_to_tree))]))

        best_action = self.actions[self.mdp.policy[current_state][0]]
        selected = []
        for i, is_selected in enumerate(reversed(best_action)):
            if is_selected == 1:
                selected.append(i)
        print("Selected: {}".format(selected))
        return selected

    def get_actions(self):
        return np.asarray([a for a in itertools.product(*[[0, 1] for _ in range(self.num_news)])
                           if sum(a) == self.budget])

    def get_transition(self):
        s = 2 ** self.num_news
        a = int(comb(self.num_news, self.budget))
        mat = np.zeros((a, s, s))
        for j in range(s):
            s_bin = i2b(j, a)
            for i in range(a):
                next_s_bin = self.actions[i] + s_bin
                if 2 in next_s_bin:  # Not reachable
                    mat[i][j][i] = 1
                else:
                    next_s = b2i(next_s_bin)
                    mat[i][j][next_s] = 1
        return mat

    def get_reward(self,
                   active_news_to_tree: Dict[Any, PropagationTree],
                   network_propagation: TwitterNetworkPropagation):
        s = 2 ** self.num_news
        a = int(comb(self.num_news, self.budget))
        mat = np.zeros((s, a))

        voting_result = []
        non_exposed_user_to_fake_news = []

        for news_info, tree in active_news_to_tree.items():
            if is_not_checked_and_not_blocked(network_propagation, news_info):

                flag_log = tree.getattr("flag_log", [])
                weighted_vote = sum(self.scaling * self.get_p_flag_fake(node_id) * self.get_p_not_flag_not_fake(node_id)
                                    for _, node_id in flag_log)
                voting_result.append((news_info, weighted_vote))

                expose_log = tree.getattr("expose_log")
                exposure_candidates = set()
                for t, node in expose_log:
                    exposure_candidates.update(network_propagation.user_id_to_follower_ids[node])
                for first_hop_candidate in deepcopy(exposure_candidates):
                    exposure_candidates.update(network_propagation.user_id_to_follower_ids[first_hop_candidate])
                non_exposed_user_to_fake_news.append(
                    (news_info, exposure_candidates.difference({exposed for t, exposed in expose_log}))
                )

        info_to_votes = dict(voting_result)
        info_to_non_exposed_user = dict(non_exposed_user_to_fake_news)

        for i in range(s):
            for j in range(a):
                if j in info_to_votes:
                    mat[i][j] = info_to_votes[j] * len(info_to_non_exposed_user[j])
        return mat

