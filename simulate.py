from NP import *
from NPExtension import *
from TestFDC import *
from Baselines import *
from termcolor import cprint
from MatplotlibUtill import *


def get_synthetic_twitter_network(num_of_trees: int,
                                  expected_num_of_nodes: int,
                                  fake_ratio: float,
                                  base_seed: int = None) -> TwitterNetworkPropagation:
    """
    :param num_of_trees: number of stories to propagate
    :param expected_num_of_nodes: num_of_nodes of each news ~ Possion(expected_num_of_nodes)
    :param fake_ratio: probs that a news is fake
    :param base_seed: fix randomness
    :return: TwitterNetworkPropagation
    """
    g_and_tree_list = []
    for i in range(num_of_trees):
        num_of_nodes = np.random.poisson(expected_num_of_nodes)
        seed_value = base_seed + i if base_seed else None

        g = nx.random_tree(num_of_nodes, seed=seed_value)
        root = sorted(nx.betweenness_centrality(g).items(), key=lambda kv: -kv[1])[0][0]
        is_fake = True if np.random.random() < fake_ratio else False
        tree = get_propagation_tree_of_full_graph(g, root, seed_value, info=i, is_fake=is_fake)

        g_and_tree_list.append((g, tree))

    twitter_network = get_twitter_network(dict(g_and_tree_list))
    for i, (_, t) in enumerate(g_and_tree_list):
        twitter_network.assign_one_propagation(i, t)

    cprint("Finished: get_synthetic_twitter_network", "green")
    cprint("\t- Fake: {}, True: {}".format(
        len(twitter_network.filter_trees(lambda tree_x: tree_x.is_fake)),
        len(twitter_network.filter_trees(lambda tree_x: not tree_x.is_fake)),
    ), "green")
    cprint("\t- num_of_trees: {}, num_of_nodes_all: {}, num_of_events: {}".format(
        num_of_trees,
        len(twitter_network.nodes),
        sum(len(tree) for tree in twitter_network.info_to_tree.values()),
    ), "green")
    return twitter_network


def get_node_to_abc(nodes: List[Any],
                    type_to_assign_probs: Dict[Tuple, float],
                    seed_value=None) -> Dict[Any, Tuple]:
    """
    - a: probability that user u would not flag the news as fake, conditioned on that news x is not fake.
    - b: probability that user u would flag the news as fake, conditioned on that news x is fake.
        - a, b are conditioned on the user is reviewing the content.
    - c: probability that user u abstains from actively reviewing the news content (does not flag the news).

    :param nodes: [n1, n2, ..., n_i, ...]
    :param type_to_assign_probs: (a, b, c) -> p (<=1)
    :param seed_value:
    :return: n_i -> (a, b, c)
    """
    np.random.seed(seed_value)

    types = list(type_to_assign_probs.keys())
    num_types = len(types)
    assign_probs = list(type_to_assign_probs.values())

    node_to_abc = dict()
    for n in nodes:
        chosen_type = types[np.random.choice(num_types, p=assign_probs)]
        node_to_abc[n] = chosen_type

    return node_to_abc


def flag_by_abc(network_propagation: TwitterNetworkPropagation, event: Event,
                node_to_abc: Dict[Any, Tuple], is_verbose: bool, seed_value=None, **kwargs):
    np.random.seed(seed_value)

    followers = network_propagation.user_id_to_follower_ids[event.node_id]
    info = event.getattr("info")
    is_fake = event.getattr("is_fake")

    flag_log: List[Tuple[int or float, Any]] = []
    expose_log: List[Tuple[int or float, Any]] = []

    for f_id in followers:

        # If the node has been already propagated, do not consider it.
        if network_propagation.get_status(info, f_id, "propagated"):
            continue

        expose_log.append((network_propagation.current_time, f_id))

        # If the node has been already flagged, this is considered as exposed node, but do not have chance to flag.
        if network_propagation.get_status(info, f_id, "flagged"):
            continue

        p_not_flag_not_fake, p_flag_fake, p_abstain = node_to_abc[f_id]

        # If the node do not abstain to flag,
        if np.random.choice([True, False], p=[1 - p_abstain, p_abstain]):
            if (is_fake and np.random.choice([True, False], p=[p_flag_fake, 1 - p_flag_fake])) or \
                    (not is_fake and np.random.choice([True, False], p=[1 - p_not_flag_not_fake, p_not_flag_not_fake])):
                network_propagation.set_status(info, f_id, "flagged", True)
                flag_log.append((network_propagation.current_time, f_id))

    tree = network_propagation.get_propagation(info)
    tree.setattr("expose_log", tree.getattr("expose_log", []) + expose_log)
    tree.setattr("flag_log", tree.getattr("flag_log", []) + flag_log)

    if is_verbose:
        print("Info [{}, {}] e{}/f{} after <{}> at {}t".format(
            info, is_fake, tree.getattr("exposed", 0), len(tree.getattr("flag_log", [])), event,
            network_propagation.current_time)
        )


def block_information_by_model(network_propagation: TwitterNetworkPropagation, event: Event,
                               model, budget: int, start_time: int or float, select_exact: bool, is_verbose: bool,
                               **kwargs):
    assert callable(getattr(model, "select_fake_news", None))

    if network_propagation.current_time < start_time:
        return

    active_news_to_tree = {i: t for i, t in network_propagation.info_to_tree.items()
                           if not network_propagation.is_blocked(i)}

    selected_fake_news: List[Any] = model.select_fake_news(
        active_news_to_tree=active_news_to_tree,
        network_propagation=network_propagation,
        budget=budget,
        select_exact=select_exact,
    )

    for f_info in selected_fake_news:
        network_propagation.block_info(f_info)
        f_tree = network_propagation.get_propagation(f_info)
        if is_verbose:
            cprint("Block Info [{}, {}] e{}/f{} at {}t".format(
                f_info, f_tree.getattr("is_fake"), f_tree.getattr("exposed", 0), f_tree.getattr("flagged", 0),
                network_propagation.current_time,
            ), "red")


def get_blocked_time_of_fake_news(finished_np: TwitterNetworkPropagation):
    assert finished_np.next_time == finished_np.get_time_to_finish()

    block_log = finished_np.block_log

    blocked_time_of_fake_news = []

    for info, tree in finished_np.info_to_tree.items():
        is_fake = tree.getattr("is_fake")
        expose_log = tree.getattr("expose_log")
        flag_log = tree.getattr("flag_log")

        blocked_time = finished_np.get_blocked_time(info)
        if blocked_time != -1 and is_fake:
            blocked_time_of_fake_news.append(blocked_time)

    return blocked_time_of_fake_news


def simulate_models(models: List, seed_value=None,
                    num_of_trees=50, expected_num_of_nodes=100, fake_ratio=0.2,
                    budget=1, start_time=3, select_exact=True,
                    is_verbose=True):
    _synthetic_network = get_synthetic_twitter_network(
        num_of_trees=num_of_trees,
        expected_num_of_nodes=expected_num_of_nodes,
        fake_ratio=fake_ratio,
        base_seed=seed_value,
    )

    global_c = 0.9
    node_to_abc_in_main = get_node_to_abc(
        nodes=_synthetic_network.nodes,
        type_to_assign_probs={
            (0.9, 0.9, global_c): 0.25,
            (0.7, 0.7, global_c): 0.25,
            (0.3, 0.3, global_c): 0.25,
            (0.1, 0.1, global_c): 0.25,
        },
        seed_value=seed_value,
    )
    _synthetic_network.add_event_listener(
        event_type="propagate_all_delta",
        scope="event",
        callback_func=flag_by_abc,
        node_to_abc=node_to_abc_in_main,
        seed_value=seed_value,
        is_verbose=is_verbose,
    )

    synthetic_network_list = []
    for model in models:
        cprint("Simulate {}".format(model.__class__.__name__), "blue")

        synthetic_network = deepcopy(_synthetic_network)
        synthetic_network.add_event_listener(
            event_type="propagate_all_delta",
            scope="propagation",
            callback_func=block_information_by_model,
            model=model,
            budget=budget,
            start_time=start_time,
            select_exact=select_exact,
            is_verbose=is_verbose,
        )
        for t in range(synthetic_network.get_time_to_finish()):
            synthetic_network.propagate_all_delta()
        synthetic_network_list.append(synthetic_network)

        cprint("End of {}".format(model.__class__.__name__), "blue")

    return synthetic_network_list


if __name__ == '__main__':
    models_to_test = [MajorityVoting(), Random()]
    finished_networks = simulate_models(models=models_to_test, seed_value=42,
                                        num_of_trees=150, expected_num_of_nodes=100, fake_ratio=0.2,
                                        budget=1, start_time=3, select_exact=True,
                                        is_verbose=False)

    num_of_fake_news = len(finished_networks[0].filter_trees(lambda x: x.is_fake))
    finished_time = max(fn.get_time_to_finish() for fn in finished_networks)

    bts = []
    for net in finished_networks:
        bt = get_blocked_time_of_fake_news(net)
        bts.append(bt)

    build_hist(bts, [m.__class__.__name__ for m in models_to_test],
               title="Blocked Fake News", xlabel="time", ylabel="number",
               range=(0, finished_time), cumulative=True, histtype='step')
