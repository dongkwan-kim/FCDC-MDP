from simulate import *


def data_generation_for_m(seed_value=None,
                          number_of_nodes=150, num_of_trees=12, propagation_prob=0.5,
                          with_draw=False, fake_ratio=0.5):
    _synthetic_network = get_synthetic_twitter_network_from_scratch(
        n=number_of_nodes, alpha=0.35, beta=0.6, gamma=0.05, delta_in=0.2, delta_out=0.5,
        force_save=False, base_seed=seed_value, with_draw=with_draw,
        number_of_props=num_of_trees, propagation_prob=propagation_prob, max_iter=2000,
    )
    assign_fake_news_label(_synthetic_network, fake_ratio=fake_ratio, seed=seed_value)

    global_c = 0
    node_to_abc_in_main = get_node_to_abc(
        nodes=_synthetic_network.nodes,
        type_to_assign_probs={
            (0.9, 0.9, global_c): 0.3,
            (0.7, 0.7, global_c): 0.25,
            (0.3, 0.3, global_c): 0.25,
            (0.1, 0.1, global_c): 0.2,
        },
        seed_value=seed_value,
    )
    _synthetic_network.add_event_listener(
        event_type="propagate_all_delta",
        scope="event",
        callback_func=flag_by_abc,
        node_to_abc=node_to_abc_in_main,
        seed_value=seed_value,
        is_verbose=False,
    )

    for t in range(_synthetic_network.get_time_to_finish()):
        _synthetic_network.propagate_all_delta()

    np.set_printoptions(linewidth=200)
    print(_synthetic_network.get_cumulative_matrix_from_log("expose_log"))
    print(_synthetic_network.get_cumulative_matrix_from_log("flag_log"))
    print([_synthetic_network.info_to_tree[info].getattr("is_fake")
           for info in range(len(_synthetic_network.info_to_tree))])


def m_get_blocked_time_of_fake_news(finished_np: TwitterNetworkPropagation, m_result: List[int]):
    assert finished_np.next_time == finished_np.get_time_to_finish()

    blocked_time_of_fake_news = []

    for info, tree in finished_np.info_to_tree.items():
        is_fake = tree.getattr("is_fake")
        if is_fake:
            blocked_time = m_result.index(info)
            blocked_time_of_fake_news.append(blocked_time)

    return blocked_time_of_fake_news


def m_get_non_exposed_user_to_fake_news(finished_np: TwitterNetworkPropagation, m_result: List[int]) -> List[List[Any]]:
    assert finished_np.next_time == finished_np.get_time_to_finish()

    non_exposed_user_to_fake_news = []

    for info, tree in finished_np.info_to_tree.items():
        is_fake = tree.getattr("is_fake")
        if is_fake and finished_np.is_blocked(info):
            expose_log = tree.getattr("expose_log")
            blocked_time = m_result.index(info)
            events_not_happened = [e for e in tree if e.time_stamp >= blocked_time]
            followers = []
            for e in events_not_happened:
                followers += finished_np.user_id_to_follower_ids[e.node_id]
            non_exposed_user_to_fake_news.append(
                list(set(followers).difference({exposed for t, exposed in expose_log}))
            )
    return non_exposed_user_to_fake_news


def use_result_from_matlab():
    results = [[11, 12, 1, 9, 8, 2, 5, 10, 6, 4, 7, 3],
               [7, 12, 8, 2, 11, 5, 4, 1, 3, 10, 9, 6],
               [6, 12, 1, 4, 7, 2, 5, 11, 3, 10, 9, 8],
               [6, 1, 12, 4, 8, 3, 11, 5, 7, 10, 9, 2],
               [12, 11, 4, 10, 9, 2, 8, 1, 7, 5, 6, 3]]

    return [[i - 1 for i in r] for r in results]


def evaluate(matlab_result, with_part_draw=True, with_main_draw=True):

    trials = 5
    num_models = 3
    stacked_bars = []
    stacked_bts = [[] for _ in range(num_models)]
    _model_names = None

    for s, m_result in zip(range(trials), matlab_result):
        seed_for_this_iter = 874132 + s

        _models_to_test = [MajorityVoting(), Random()]
        _finished_networks = simulate_models(models=_models_to_test, seed_value=seed_for_this_iter,
                                             number_of_nodes=150, num_of_trees=12, propagation_prob=0.5,
                                             fake_ratio=0.5, with_draw=False,
                                             budget=1, start_time=0, select_exact=True,
                                             is_verbose=False)

        _finished_time = max(fn.get_time_to_finish() for fn in _finished_networks)
        _model_names = ["MDP"] + [m.__class__.__name__ for m in _models_to_test]

        _neu = [m_get_non_exposed_user_to_fake_news(_finished_networks[0], m_result)]
        _neu += [get_non_exposed_user_to_fake_news(net) for net in _finished_networks]
        bars = [sum([len(u) for u in us]) for us in _neu]
        if with_part_draw:
            build_bar(_model_names, bars, ylabel="# of Users", title="Non-exposed Users to Fake News - {}".format(s))
        stacked_bars.append(bars)

        _bts = [m_get_blocked_time_of_fake_news(_finished_networks[0], m_result)]
        _bts += [get_blocked_time_of_fake_news(net) for net in _finished_networks]
        if with_part_draw:
            build_hist(_bts, _model_names,
                       title="Blocked Fake News - {}".format(s), xlabel="time", ylabel="number",
                       range=(0, _finished_time), cumulative=True, histtype='step')
        stacked_bts = [a+b for a, b in zip(stacked_bts, _bts)]

    stacked_bars = np.asarray(stacked_bars)
    t_stacked_bars = np.transpose(stacked_bars)
    mean_bars = [np.mean(bars) for bars in t_stacked_bars]
    stdev_bars = [np.std(bars) for bars in t_stacked_bars]

    if with_main_draw:
        build_bar(_model_names, mean_bars,
                  ylabel="# of Users", title="Mean # of Users Not Exposed to Fake News", yerr=stdev_bars)

        build_hist(stacked_bts, _model_names,
                   title="Blocked Fake News", xlabel="time", ylabel="number",
                   cumulative=True, histtype='step')


if __name__ == '__main__':
    evaluate(use_result_from_matlab())
