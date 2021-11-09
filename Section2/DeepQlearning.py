"""
INHERTES FROM SECTION 1 CLASS REIMPLEMENTING THE Q FUNCTION
Neural Network: the network should take a state as an input (or a minibatch
of states) and output the predicted q-value of each action for that state. The
network will be trained by minimizing the loss function with an
optimization method of your choice (SGD, RMSProp, …). Try 2 different
structures for the network, the first with 3 hidden layers and the second
with 5.
Notice: the network will be used both as the q-value network and as the
target network, but with different weights.
- experience_replay: a deque in a fixed size to store past experiences.
- sample_batch: sample a minibatch randomly from the experience_replay.
- sample_action: choose an action with decaying 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦 method or
another method of your choice.
- train_agent: train the agent with the algorithm detailed in figure 4.
- test_agent: test the agent on a new episode with the trained model. You can
render the environment if you want to watch your agent play.
"""
import os
import numpy as np
from typing import *
from collections import deque
import tensorflow as tf
import tensorboard


class ExperienceReplayDeque:
    def __init__(self, **kwargs):
        self.max_size = kwargs.get('max_deque_size', 100)
        self._deque = []

    def get_entire_deque(self):
        return self._deque.copy()

    def __len__(self):
        return len(self._deque)

    def append(self, experiences: Union[Dict, List, Set, Tuple, List[Union[Dict,List, Tuple, Set]]]):
        """
        experiences can be either dictionary (with the keys 'state', 'action' and 'reward') or list/tuple/set the size of 3
        with the same order of values (i.e. x[0] = state, x[1]= action etc.)
        FOR BATCH APPENDING:
            experiences must be a list where each element obeys to the above constraints.
        :param experiences:
        :return:
        """
        # for batch appending
        if isinstance(experiences, list) and isinstance(experiences[0], Sequence):
            for single_experience in experiences:
                self.append(single_experience)
            return

        to_append = None
        if len(experiences) != 3:
            raise ValueError(f'Each experience must be size 3: (state, action, reward) triplet but x is {experiences}')

        if isinstance(experiences, dict):
            to_append = [experiences['state'], experiences['action'], experiences['reward']]
        else:
            to_append = [experiences[0], experiences[1], experiences[2]]

        if len(self._deque)+1 > self.max_size:
            Warning('Out of room in deque, removing the first experience to make room')
            self._deque.pop(0)

        self._deque.append(to_append)

    def remove_samples(self, indices: Union[int, List[int]]):
        if isinstance(indices, Sequence):
            for idx in indices:
                self.remove_samples(idx)
            return

        self._deque.pop(indices)

    def pop_samples(self, num_of_samples: int = 1):
        if len(self._deque) == 0:
            raise ValueError("No experiences in deque")
        elif num_of_samples >= len(self._deque):
            Warning(f'Not enough samples in deque, returing the maximum possible: {len(self._deque)} samples')
            samples_indices_to_return = np.random.randint(0, len(self._deque), size=len(self._deque), dtype=np.int)
            samples_to_return = np.array(self._deque)[samples_indices_to_return]
        else:
            samples_indices_to_return = np.random.randint(0, len(self._deque), size=num_of_samples, dtype=int)
            samples_to_return = np.array(self._deque)[samples_indices_to_return]

        return samples_to_return


class NeuralQEstimator:
    def __init__(self, **kwargs):
        self.network_layer_structure = kwargs.get('layers_structure', (100, 50, 20))



def sample_batch():
    pass


def sample_action():
    pass


def train_agent():
    pass


def test_agent():
    pass



if __name__ == '__main__':
    # briefly testing the experience replay deque
    erd = ExperienceReplayDeque(max_deque_size=2)
    exp_dict = {'state': 2, 'action': 0, 'reward': 12}
    exp_lst = [1, 2, 3]
    # should raise an exception
    try:
        erd.append({'state': 2, 'action': 0})
    except ValueError as e:
        pass

    erd.append(exp_dict)
    erd.append(exp_lst)
    print(erd.get_entire_deque())
    assert len(erd) == 2
    # should present a warning that the deque is overflowing and that the first experience entered was removed
    erd.append({'state': 11, 'action': 22, 'reward': 33})
    assert len(erd) == 2

    print(erd.get_entire_deque())

    print(erd.pop_samples(1))
    print(erd.pop_samples(1))
    # should raise a value error
    try:
        print(erd.pop_samples(1))
    except ValueError as e:
        pass

    # batch appending
    erd = ExperienceReplayDeque(max_deque_size=100)
    rand_experiences = np.random.randint(0, 200, (200, 3))
    erd.append(rand_experiences.tolist())
    assert len(erd) == 100

    random_samples = erd.pop_samples(50)
    print(random_samples)
    assert len(random_samples) == 50
    assert len(erd) == 100

    erd.remove_samples([x for x in range(50)])
    assert len(erd) == 50
