from typing import Callable, List, Any

import numpy as np


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
        else:
            self.reward_func = self.get_default_reward_func()

        self.transitions, self.transition_func = None, None
        if isinstance(transitions, list):
            self.transitions = np.array(transitions)
            assert self.transitions.shape == (len(states), len(actions), len(states))
        elif callable(transitions):
            self.transition_func = transitions
        else:
            self.transition_func = self.get_default_transition_func()

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

    @classmethod
    def get_default_reward_func(cls) -> Callable:
        raise NotImplementedError

    @classmethod
    def get_default_transition_func(cls) -> Callable:
        raise NotImplementedError


class FCDCMDP(MDP):

    def __init__(self, states: List[Any], actions: List[Any],
                 rewards: List or Callable = None, transitions: List or Callable = None,
                 discount: float = 0.95, epsilon: float = 0.01):
        super().__init__(states, actions, rewards, transitions, discount, epsilon)

    def update_states(self, new_states: List[Any]):
        self.states = np.concatenate(self.states, np.array(new_states))

    @classmethod
    def get_default_reward_func(cls) -> Callable:

        def default_reward_func(state, action) -> float:
            raise NotImplementedError

        return default_reward_func

    @classmethod
    def get_default_transition_func(cls) -> Callable:

        def default_transition_func(state, action) -> Any:
            raise NotImplementedError

        return default_transition_func
