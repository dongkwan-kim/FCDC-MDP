from typing import Callable, List, Any, Dict

import numpy as np

from NP import PropagationTree
from NPExtension import TwitterNetworkPropagation


class MDP:

    def __init__(self, states: List[Any], actions: List[Any],
                 rewards: List or Callable = None, transitions: List or Callable = None,
                 discount: float = 0.95, epsilon: float = 0.01):
        """
        :param states: list
        :param actions: list
        :param rewards: list of shape (S, A) or Callable
        :param transitions: list of shape (S, A, S) or Callable
        :param discount: float
        :param epsilon: float
        """

        self.states = np.array(states)
        self.actions = np.array(actions)

        self.rewards, self.reward_func = None, None
        if isinstance(rewards, list):
            self.rewards = np.array(rewards)
            assert self.rewards.shape == (len(states), len(actions))
        elif callable(rewards):
            self.reward_func = rewards

        self.transitions, self.transition_func = None, None
        if isinstance(transitions, list):
            self.transitions = np.array(transitions)
            assert self.transitions.shape == (len(states), len(actions), len(states))
        elif callable(transitions):
            self.transition_func = transitions

        self.discount = discount
        self.epsilon = epsilon

        self.values = None
        self.policies = None

    def tack_action(self, state, action) -> Any:
        if self.transition_func:
            return self.transition_func(state, action)
        elif self.transitions:
            return np.random.choice(self.states, p=self.transitions[state][action][:])

    def get_reward(self, state, action) -> float:
        if self.reward_func:
            return self.reward_func(state, action)
        elif self.rewards:
            return self.rewards[state][action]


class FCDCMDP(MDP):

    def __init__(self, states: List[Any], actions: List[Any],
                 discount: float = 0.95, epsilon: float = 0.01):
        rewards = self.fcdc_reward
        transitions = self.fcdc_transition
        super().__init__(states, actions, rewards, transitions, discount, epsilon)

    def select_fake_news(self,
                         active_news_to_tree: Dict[Any, PropagationTree],
                         network_propagation: TwitterNetworkPropagation,
                         budget: int,
                         select_exact: bool) -> List[Any]:

        for news_info, tree in active_news_to_tree.items():
            flag_log = tree.getattr("flag_log", [])
            expose_log = tree.getattr("expose_log", [])

        return []

    def get_prob_being_fake(self, flag_log, expose_log):
        pass


    def get_reward_of_adding_one(self, e, f):
        return


    def fcdc_transition(self, state, action):
        pass
