from NPExtension import *


class Random:

    def __init__(self):
        pass

    def select_fake_news(self,
                         active_news_to_tree: Dict[Any, PropagationTree],
                         network_propagation: TwitterNetworkPropagation,
                         budget: int,
                         select_exact: bool) -> List[Any]:
        news_infos = [news_info for news_info in active_news_to_tree.keys()]
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
            flagged = tree.getattr("flagged")
            voting_result.append((news_info, flagged))

        voting_result = sorted(voting_result, key=lambda nt: -nt[1])
        return [news_info for news_info, flagged in voting_result[:budget]]
