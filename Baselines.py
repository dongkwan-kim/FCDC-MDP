from functools import reduce

from NPExtension import *


class Random:

    def __init__(self):
        pass

    def select_fake_news(self,
                         active_news_to_tree: Dict[Any, PropagationTree],
                         network_propagation: TwitterNetworkPropagation,
                         budget: int,
                         select_exact: bool) -> List[Any]:
        news_infos = [news_info for news_info in active_news_to_tree.keys()
                      if not network_propagation.is_blocked(news_info)]
        np.random.shuffle(news_infos)
        return news_infos[:budget]


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
            if not network_propagation.is_blocked(news_info):
                flagged = len(tree.getattr("flag_log", []))
                voting_result.append((news_info, flagged))

        voting_result = sorted(voting_result, key=lambda nt: -nt[1])
        return [news_info for news_info, flagged in voting_result[:budget]]


class WeightedUserModel:

    def __init__(self):
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


class WeightedMajorityVoting(WeightedUserModel):

    def __init__(self):
        super().__init__()
        self.scaling = 4

    def select_fake_news(self,
                         active_news_to_tree: Dict[Any, PropagationTree],
                         network_propagation: TwitterNetworkPropagation,
                         budget: int,
                         select_exact: bool) -> List[Any]:

        voting_result: List[Tuple] = []
        for info, tree in active_news_to_tree.items():
            if not network_propagation.is_blocked(info):
                flag_log = tree.getattr("flag_log", [])
                weighted_vote = sum(self.scaling * self.get_p_flag_fake(node_id) * self.get_p_not_flag_not_fake(node_id)
                                    for _, node_id in flag_log)
                voting_result.append((info, weighted_vote))

        voting_result = sorted(voting_result, key=lambda nt: -nt[1])
        selected_news = [news_info for news_info, flagged in voting_result[:budget]]

        for sn_info in selected_news:
            tree = active_news_to_tree[sn_info]
            self.inc_node_history(sn_info, is_node_flagged=True, is_real_fake=tree.getattr("is_fake"))

        return selected_news
