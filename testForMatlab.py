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

    print(_synthetic_network.get_cumulative_matrix_from_log("expose_log"))
    print(_synthetic_network.get_cumulative_matrix_from_log("flag_log"))
    print([_synthetic_network.info_to_tree[info].getattr("is_fake")
           for info in range(len(_synthetic_network.info_to_tree))])


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)
    for s in range(5):
        data_generation_for_m(seed_value=874132+s)
