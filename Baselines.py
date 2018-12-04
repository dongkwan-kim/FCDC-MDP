from functools import reduce

from NPExtension import *


def is_not_checked_and_not_blocked(network_propagation: TwitterNetworkPropagation, news_info):
    check_log = network_propagation.getattr("check_log", [])
    is_not_blocked = (not network_propagation.is_blocked(news_info))
    is_not_checked = (news_info not in check_log)
    return is_not_blocked and is_not_checked


class Random:

    def __init__(self):
        pass

    def select_fake_news(self,
                         active_news_to_tree: Dict[Any, PropagationTree],
                         network_propagation: TwitterNetworkPropagation,
                         budget: int,
                         select_exact: bool) -> List[Any]:
        news_infos = [news_info for news_info in active_news_to_tree.keys()
                      if is_not_checked_and_not_blocked(network_propagation, news_info)]
        np.random.shuffle(news_infos)

        selected = news_infos[:budget]
        network_propagation.setattr("check_log", network_propagation.getattr("check_log", []) + selected)
        return selected


class MajorityVoting:

    def __init__(self):
        pass

    def select_fake_news(self,
                         active_news_to_tree: Dict[Any, PropagationTree],
                         network_propagation: TwitterNetworkPropagation,
                         budget: int,
                         select_exact: bool) -> List[Any]:
        voting_result: List[Tuple] = []
        for news_info, tree in active_news_to_tree.items():
            if is_not_checked_and_not_blocked(network_propagation, news_info):
                flagged = len(tree.getattr("flag_log", []))
                voting_result.append((news_info, flagged))

        voting_result = sorted(voting_result, key=lambda nt: -nt[1])

        selected = [news_info for news_info, flagged in voting_result[:budget]]
        network_propagation.setattr("check_log", network_propagation.getattr("check_log", []) + selected)
        return selected


class WeightedUserModel:

    def __init__(self):
        self.scaling = 1
        self.node_to_history = defaultdict(lambda: [[0, 0], [0, 0]])

    def get_p_not_flag_not_fake(self, node):
        not_flag_not_fake, flag_not_fake = self.node_to_history[node][0]
        s = not_flag_not_fake + flag_not_fake
        # Prior = x(1-x), Likelihood = x^p (1-x)^q
        return (not_flag_not_fake + 1) / (s + 2)

    def get_p_flag_fake(self, node):
        not_flag_fake, flag_fake = self.node_to_history[node][1]
        s = not_flag_fake + flag_fake
        # Prior = x(1-x), Likelihood = x^p (1-x)^q
        return (flag_fake + 1) / (s + 2)

    def inc_node_history(self, node, is_node_flagged, is_real_fake):
        idx_real_fake = 0 if not is_real_fake else 1
        idx_node_flagged = 0 if not is_node_flagged else 1
        self.node_to_history[node][idx_real_fake][idx_node_flagged] += 1


class WeightedMV(WeightedUserModel):

    def __init__(self):
        super().__init__()
        self.scaling = 4

    def select_fake_news(self,
                         active_news_to_tree: Dict[Any, PropagationTree],
                         network_propagation: TwitterNetworkPropagation,
                         budget: int,
                         select_exact: bool) -> List[Any]:

        voting_result: List[Tuple] = []
        for news_info, tree in active_news_to_tree.items():
            if is_not_checked_and_not_blocked(network_propagation, news_info):
                flag_log = tree.getattr("flag_log", [])
                weighted_vote = sum(self.scaling * self.get_p_flag_fake(node_id) * self.get_p_not_flag_not_fake(node_id)
                                    for _, node_id in flag_log)
                voting_result.append((news_info, weighted_vote))

        voting_result = sorted(voting_result, key=lambda nt: -nt[1])
        selected = [news_info for news_info, flagged in voting_result[:budget]]

        # Update users' history
        for sn_info in selected:
            tree = active_news_to_tree[sn_info]
            self.inc_node_history(sn_info, is_node_flagged=True, is_real_fake=tree.getattr("is_fake"))

        network_propagation.setattr("check_log", network_propagation.getattr("check_log", []) + selected)
        return selected
