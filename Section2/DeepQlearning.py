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
import datetime
import os
import shutil
import copy
import random
import re
import keras.callbacks
import numpy as np
from typing import *
import tensorflow as tf
import gym
from Section1.BaseQlearning import BaseQlearningAgent
from Section1.visualization import plot_reward_per_episode, plot_average_num_of_steps_to_reach_goal
from tensorflow.keras.callbacks import TensorBoard


ENVIRONMENT = gym.make("CartPole-v1")

STATE_SPACE_SIZE = ENVIRONMENT.observation_space.shape[0]
ACTION_SPACE_SIZE = ENVIRONMENT.action_space.n

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
        if len(experiences) != 4:
            raise ValueError(f'Each experience must be size 3: (state, action, reward) triplet but x is {experiences}')

        if isinstance(experiences, dict):
            to_append = [experiences['state'], experiences['action'], experiences['reward'], experiences['is_goal']]
        else:
            to_append = [experiences[0], experiences[1], experiences[2], experiences[3]]

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
            samples_indices_to_return = np.random.randint(0, len(self._deque), size=len(self._deque), dtype=int)
            samples_to_return = np.array(self._deque)[samples_indices_to_return]
        else:
            samples_indices_to_return = np.random.randint(0, len(self._deque), size=num_of_samples, dtype=int)
            samples_to_return = np.array(self._deque)[samples_indices_to_return]

        return samples_to_return


class NeuralQEstimator:
    def __init__(self, **kwargs):
        """
        defaultive values are according to the gym cartpole_v1 environment
        :param kwargs:
        """
        self.network_layer_structure = kwargs.get('layers_structure', (100, 50, 20))
        self.dropout_rate = kwargs.get('dropout_rate', 0.5)
        self.layers_activation_func = kwargs.get('activation_function', tf.nn.relu)
        self.network_optimizer = kwargs.get('network_optimizer', tf.optimizers.RMSprop)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.network_optimizer = self.network_optimizer(lr=learning_rate)
        self.network_loss_function = kwargs.get('network_loss', 'mse')
        self.action_space_size = kwargs.get('action_space_size', ACTION_SPACE_SIZE)
        self.state_space_size = kwargs.get('state_space_size', STATE_SPACE_SIZE)
        self.random_action_probability = kwargs.get('epsilon_greedy', 0.001)
        # # experience replay data structure
        # self.experience_replay_max_size = kwargs.get('experience_replay_max_size', 100)
        # self.network_experience_replay = ExperienceReplayDeque(max_deque_size=self.experience_replay_max_size)
        #

        # add input layer
        layers = []
        input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_space_size))
        # adding all layers
        layers.append(input_layer)
        for layer_idx, layer_size in enumerate(self.network_layer_structure):
            if isinstance(layer_size, str) == 'dropout':
                layers.append(tf.keras.layers.Dropout(self.dropout_rate, name=f'layer_{layer_idx}_dropout'))
                continue
            layers.append(tf.keras.layers.Dense(layer_size, activation=self.layers_activation_func, name=f'layer_{layer_idx}'))

        # adding output layer
        layers.append(tf.keras.layers.Dense(self.action_space_size, activation='relu'))

        # creating the model
        self.model = tf.keras.Sequential(layers)

        self.model.compile(self.network_optimizer, loss=self.network_loss_function)

    def predict(self, input):
        assert input.shape[1] == self.state_space_size, f'input 2nd dimension must be {self.state_space_size}'
        return self.model.predict(input)

    def fit(self, x, y, callbacks=None, **kwargs):
        assert x.shape[1] == self.state_space_size, f'x 2nd dimension must be {self.state_space_size}'
        assert y.shape[1] == self.action_space_size, f'y output 2nd dimension must be {self.action_space_size}'
        return self.model.fit(x, y, callbacks=callbacks, **kwargs)

    def update_model(self, model):
        self.model = model

    def summary(self):
        return self.model.summary()


class StepsCountMetric(tf.keras.metrics.Metric):
    """
    custom metric to keep track of #steps for each accumulated reward
    """
    def __init__(self):
        super(StepsCountMetric, self).__init__(name='train_steps_num', dtype=tf.int32)
        self.val = None

    def update_state(self, x):
        self.val = x

    def result(self):
        return self.val


class LossCustomMetric(tf.keras.metrics.Metric):
    """
    custom metric to keep track of loss in each training step
    """
    def __init__(self):
        super(LossCustomMetric, self).__init__(name='training_step_loss', dtype=tf.int32)
        self.val = None

    def update_state(self, x):
        self.val = x

    def result(self):
        return self.val


class DeepQLearner(BaseQlearningAgent):
    def __init__(self,
                 enviorment,
                 goal_state,
                 num_episods=5000,
                 max_steps_per_episode=100,
                 learning_rate=0.01,
                 learning_rate_decay=0.99,
                 discount_rate=0.99,
                 expolaration_decay_rate=0.001,
                 min_expolaration_rate=0.01,
                 **kwargs) -> None:

        super(DeepQLearner, self).__init__(enviorment,goal_state,num_episods,max_steps_per_episode,learning_rate,learning_rate_decay,discount_rate,expolaration_decay_rate, min_expolaration_rate)
        self.exp_replay_size = kwargs.get('max_experience_replay_size', 5000)
        self.experience_replay_queue = ExperienceReplayDeque(max_deque_size=self.exp_replay_size)
        self.state_space_size = self.enviorment.observation_space.shape[0]
        self.action_space_size = self.enviorment.action_space.n
        kwargs['action_space_size'] = self.action_space_size
        kwargs['state_space_size'] = self.state_space_size
        self.nn_target = NeuralQEstimator(**kwargs)
        self.nn_q_value = NeuralQEstimator(**kwargs)
        self.number_of_steps_to_update_weights = kwargs.get('number_of_steps_to_update_weights', kwargs.get('c', 10))
        self.experience_replay_sample_size = kwargs.get('experience_replay_sample_size', 10)
        # overriding the baseQlearning exploration rate attribute
        self.expolration_rate = kwargs.get('initial_exploration_rate', 0.5)

        self.is_model_trained = False

        # for tensorboard
        self.train_summary_writer = tf.summary.create_file_writer(kwargs.get('tensor_board_train_path',
                                                                             DIR_FOR_TF_BOARD_TRAIN_LOGS))
        self.test_summary_writer = tf.summary.create_file_writer(kwargs.get('tensor_board_test_path',
                                                                            DIR_FOR_TF_BOARD_TEST_LOGS))
        self.train_episode_reward_metric = tf.keras.metrics.Mean('train_reward', dtype=tf.float32)
        self.train_steps_num_metric = tf.keras.metrics.Mean(name='train_steps_num', dtype=tf.float32)
        self.train_steps_loss_metric = LossCustomMetric()

    #     rendering flags
        self.render_during_training = kwargs.get('render_training_flag', False)
        self.render_during_testing = kwargs.get('render_testing_flag', False)

    @staticmethod
    def lr_decay_scheduler_wrapper(lr_decay):
        def lr_decay_scheduler(episode_num, lr):
            """
            custom scheduler lr decay
            :param episode_num:
            :param lr:
            :return:
            """
            return lr * lr_decay
        return lr_decay_scheduler

    def train(self):
        journy_q_tables = []
        rewards_per_episode = []
        steps_per_100_episodes = np.zeros(self.num_episods//100)
        loss_ctr = 0
        for episode_num in range(self.num_episods):
            state = self.enviorment.reset()
            episode_rewards = 0
            consecutive_episodes_with_high_reward = 0
            step_num = 0
            is_goal = False
            states_for_q_eval_updates = []
            rewards_for_q_eval_updates = []
            accumulated_reward = 0
            while step_num < self.max_steps_per_episode and not is_goal:
                loss_ctr += 1
                # todo: remove rendering before submission:
                if self.render_during_training:
                    self.enviorment.render()
                ##################

                action = self.sample_action(state.reshape(1, -1))
                new_state, reward, is_goal, info = self.enviorment.step(action)
                # episode_rewards+=reward
                accumulated_reward += reward
                # adding to experience replay
                self.experience_replay_queue.append((state, action, reward, is_goal))

                if is_goal:
                    self.train_episode_reward_metric(accumulated_reward)
                    self.train_steps_num_metric(step_num)
                    rewards_for_q_eval_updates.append(accumulated_reward)
                    print('LOST!')
                    break

                # states_for_q_eval_updates.append(state)
                # rewards_for_q_eval_updates.append(reward)

                # update the weights of the q_evaluator network with samples from experience replay
                mini_batch_from_experience_replay_raw = self.experience_replay_queue.pop_samples(self.experience_replay_sample_size)

                mini_batch_samples_predictions = np.zeros((mini_batch_from_experience_replay_raw.shape[0], self.action_space_size))
                mini_batch_samples_states = np.zeros((mini_batch_from_experience_replay_raw.shape[0], self.state_space_size))

                # give all chosen actions the reward
                action_indexes = mini_batch_from_experience_replay_raw[:, 1].astype(int)
                for i in range(mini_batch_from_experience_replay_raw.shape[0]):
                    mini_batch_samples_predictions[i][action_indexes[i]] = \
                        mini_batch_from_experience_replay_raw[i, 2]
                # update all actions' rewards (Which did not end the game) with the discounted prediction
                non_terminal_states_indexes = mini_batch_from_experience_replay_raw[:, -1] == False
                non_terminal_actions_indexes = action_indexes[non_terminal_states_indexes]
                mini_batch_samples_predictions[non_terminal_states_indexes,
                                               non_terminal_actions_indexes] += \
                    self.discount_rate * self.nn_target.predict(np.stack(mini_batch_from_experience_replay_raw[mini_batch_from_experience_replay_raw[:, -1] == False][:, 0])).max(axis=1)

                step_loss = self.nn_q_value.fit(mini_batch_samples_states,
                                    np.array(mini_batch_samples_predictions).reshape(len(mini_batch_samples_predictions),
                                                                                     self.action_space_size),
                                    callbacks=[tf.keras.callbacks.LearningRateScheduler(self.lr_decay_scheduler_wrapper(self.learning_rate_decay))],
                                    )

                self.train_steps_loss_metric(step_loss.history['loss'][0])
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss_at_training_step', self.train_steps_loss_metric.result(), step=loss_ctr)
                    self.train_steps_loss_metric.reset_states()

                if step_num % self.number_of_steps_to_update_weights == 0:
                    self.update_q()
                state = new_state

                step_num += 1

            with self.train_summary_writer.as_default():
                tf.summary.scalar('avg_reward', self.train_episode_reward_metric.result(), step=episode_num)
                tf.summary.scalar('number_of_steps_to_reward', self.train_steps_num_metric.result(), step=episode_num)
                # tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

            template = '#Episode: {}, Accumulated Average Reward: {}, Average #Steps/Episode: {}'
            print(template.format(episode_num + 1, self.train_episode_reward_metric.result(), self.train_steps_num_metric.result()))

            ## Metrics
            ## Reward per episode
            rewards_per_episode.append(accumulated_reward)

            if accumulated_reward > 475:
                consecutive_episodes_with_high_reward += 1
            else:
                consecutive_episodes_with_high_reward = 0

            if consecutive_episodes_with_high_reward == 100:
                with open('simple_stats_file.txt', 'a+') as f:
                    f.write(f'{datetime.date}|{datetime.time}: '
                            f'Number of episodes until 100 consecutive '
                            f'episode with avg reward higher than 475 is {episode_num}')

            counted_step = step_num
            if (step_num < self.max_steps_per_episode and not is_goal):
                counted_step = self.max_steps_per_episode

            ##accumulate step per 100 episodes
            steps_per_100_episodes[episode_num // 100] += counted_step / 100

            self.expolration_rate = max((self.min_expolaration_rate, self.expolration_rate * np.exp(
                -self.expolaration_decay_rate * episode_num)))
            self.learning_rate = self.learning_rate * self.learning_rate_decay
        # todo: we probably need to append here the networks' statuses.
        # journy_q_tables.append(self.q_table)

        self.train_episode_reward_metric.reset_states()
        self.train_steps_num_metric.reset_states()

        steps_per_100_episodes = steps_per_100_episodes/100

        self.train_summary(journy_q_tables, rewards_per_episode, steps_per_100_episodes)

        self.is_model_trained = True

    def sample_action(self, state):
        assert state.shape[1] == self.state_space_size, f'input 2nd dimension must be {self.state_space_size}'

        epsilon = random.uniform(0, 1)
        if epsilon > self.expolration_rate:
            action = np.argmax(self.nn_q_value.predict(state))
        else:
            action = self.enviorment.action_space.sample()

        return action

    def update_q(self, state=None, action=None, reward=None, new_state=None):
        self.nn_target.model = tf.keras.models.clone_model(self.nn_q_value.model)

    def train_summary(self, journy_q_tables, rewards_per_episode, steps_per_100_episodes):
        plot_reward_per_episode(rewards_per_episode)
        plot_average_num_of_steps_to_reach_goal(steps_per_100_episodes)
        # q_value_table_color_map(journy_q_tables[0],500,np.arange(15),['LEFT','DOWN','RIGHT','UP'])
        return

    def move(self, state):
        action = np.argmax(self.nn_target.predict(state))
        return self.enviorment.step(action)

    def play(self):
        # todo: remove before submission
        import playsound
        from time import sleep, time
        ###########
        assert self.is_model_trained, f'Model must be trained first!'
        state = self.enviorment.reset()

        # todo: remove before submission
        print(f"{'#'*10}\nSTARTED PLAYING\n{'#'*10}")
        # wait for 3 seconds
        sleep(3)
        ###########

        # playsound.playsound('C:\\Users\\User\\PycharmProjects\\DRL-A1-DQN-\\Miscellaneous\\MV27TES-alarm.mp3')
        step = 1
        if self.render_during_testing:
            self.enviorment.render()
        state, reward, is_goal, info = self.move(state.reshape(1, -1))
        seconds_remaining_stable = time()
        while (not is_goal):
            print("Step # {}:".format(step))
            self.enviorment.render()
            state, reward, is_goal, info = self.move(state.reshape(1, -1))
            step += 1
        seconds_remaining_stable = time() - seconds_remaining_stable
        # todo: figure out if there is a goal state
        if (is_goal):
            print("LOSE! # of seconds remaining stable is:{}".format(seconds_remaining_stable))
        else:
            print("NO LOSE! # of seconds is:{}".format(seconds_remaining_stable))
        return None


learning_rate = 0.01
learning_rate_decay = 1
discount_rate = 0.99
expolaration_decay_rate = 0.001
initial_exploration_rate = 0.5
min_expolaration_rate = 0.05
layers_structure = (64, 32, 16)
# network_optimizer = tf.optimizers.RMSprop(learning_rate)
network_optimizer = tf.optimizers.Adam
epsilon_greedy = 0.1
num_episods = 5000
max_steps_per_episode = 1000
number_of_steps_to_update_weights = 4

q_estimator_kwargs = {
    'layers_structure': layers_structure,
    'network_optimizer': network_optimizer,
    'epsilon_greedy': epsilon_greedy,
    'learning_rate': learning_rate,
    'learning_rate_decay': learning_rate_decay,
    'discount_rate': discount_rate,
    'expolaration_decay_rate': expolaration_decay_rate,
    'min_expolaration_rate': min_expolaration_rate,
    'num_episods': num_episods,
    'max_steps_per_episode': max_steps_per_episode,
    'number_of_steps_to_update_weights': number_of_steps_to_update_weights,
    # 'initial_exploration_rate': initial_exploration_rate,
}

regex = re.compile(r'\W*')
#First parameter is the replacement, second parameter is your input string
# todo: what are the rest of the properties to distinguish different models
kwargs_name = '_'.join([f'{regex.sub("", str(key))}={regex.sub("", str(val))}' if not type(val) ==  type(tf.optimizers.Optimizer) else regex.sub("",str(type(val)))[23:] for key, val in q_estimator_kwargs.items()])[:100]

DIR_FOR_TF_BOARD_TRAIN_LOGS = os.sep.join([os.getcwd(), f'tb_callback_dir', kwargs_name, 'train'])
DIR_FOR_TF_BOARD_TEST_LOGS = os.sep.join([os.getcwd(), f'tb_callback_dir', kwargs_name, 'test'])

q_estimator_kwargs.update(
    {
        'tensor_board_train_path': DIR_FOR_TF_BOARD_TRAIN_LOGS,
        'tensor_board_test_path': DIR_FOR_TF_BOARD_TEST_LOGS
    }
)

if os.path.isdir(DIR_FOR_TF_BOARD_TRAIN_LOGS):
    shutil.rmtree(DIR_FOR_TF_BOARD_TRAIN_LOGS, ignore_errors=True)
if os.path.isdir(DIR_FOR_TF_BOARD_TEST_LOGS):
    shutil.rmtree(DIR_FOR_TF_BOARD_TEST_LOGS, ignore_errors=True)

# ENVIRONMENT.render()
nn_q_learner = DeepQLearner(
    enviorment=ENVIRONMENT,
    goal_state=None,
    render_training_flag=True,
    render_testing_flag=True,
    **q_estimator_kwargs
)


def sample_batch():
    """
    sample a minibatch randomly from the experience_replay.
    :return:
    """
    global nn_q_learner
    experience_memory = nn_q_learner.experience_replay_queue
    return experience_memory.pop_samples()


def sample_action(state):
    """
    choose an action with decaying 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦 method or
    another method of your choice.
    :return:
    """
    global nn_q_learner
    assert state.shape[1] == STATE_SPACE_SIZE, f"State must be the correct size, expecting {state.shape}, but got shape {state.shape}"

    chosen_action = nn_q_learner.sample_action(state)

    return chosen_action


def train_agent():
    """
    train the agent with the algorithm detailed in figure 4.
    :return:
    """
    global nn_q_learner
    # train_writer = tf.summary.create_file_writer("Logs\\train")
    nn_q_learner.train()
    # experience_replay = ExperienceReplayDeque(max_deque_size=exp_replay_size)


def test_agent():
    """
    test the agent on a new episode with the trained model. You can
    render the environment if you want to watch your agent play.
    :return:
    """
    global nn_q_learner
    test_writer = tf.summary.create_file_writer("Logs\\test")
    assert nn_q_learner.is_model_trained, "Model must be trained before testing!"

    nn_q_learner.play()




if __name__ == '__main__':
    # # briefly testing the experience replay deque
    # erd = ExperienceReplayDeque(max_deque_size=2)
    # exp_dict = {'state': 2, 'action': 0, 'reward': 12}
    # exp_lst = [1, 2, 3]
    # # should raise an exception
    # try:
    #     erd.append({'state': 2, 'action': 0})
    # except ValueError as e:
    #     pass
    #
    # erd.append(exp_dict)
    # erd.append(exp_lst)
    # print(erd.get_entire_deque())
    # assert len(erd) == 2
    # # should present a warning that the deque is overflowing and that the first experience entered was removed
    # erd.append({'state': 11, 'action': 22, 'reward': 33})
    # assert len(erd) == 2
    #
    # print(erd.get_entire_deque())
    #
    # print(erd.pop_samples(1))
    # print(erd.pop_samples(1))
    # # should raise a value error
    # try:
    #     print(erd.pop_samples(1))
    # except ValueError as e:
    #     pass
    #
    # # batch appending
    # erd = ExperienceReplayDeque(max_deque_size=100)
    # rand_experiences = np.random.randint(0, 200, (200, 3))
    # erd.append(rand_experiences.tolist())
    # assert len(erd) == 100
    #
    # random_samples = erd.pop_samples(50)
    # print(random_samples)
    # assert len(random_samples) == 50
    # assert len(erd) == 100
    #
    # erd.remove_samples([x for x in range(50)])
    # assert len(erd) == 50
    # # testing nn functionality:
    # random_state = np.random.random(size=4).reshape(1, -1)
    # print(sample_action(random_state))
    train_agent()
    test_agent()
