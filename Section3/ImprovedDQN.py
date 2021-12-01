"""
TBD:
can be either dueling/double DQN or any other improvement

https://ieeexplore.ieee.org/abstract/document/8721655
"""

import datetime
import os
import shutil
import copy
import random
from typing import *
import re
import itertools
import numpy as np
import keras.callbacks
import tensorflow as tf
import gym
from Section1.BaseQlearning import BaseQlearningAgent
from Section1.visualization import plot_reward_per_episode, plot_average_num_of_steps_to_reach_goal
from Section2.DeepQlearning import NeuralQEstimator
from tensorflow.keras.callbacks import TensorBoard


# gym.envs.register(id='CartPole-v1',
#                   entry_point='gym.envs:CartPole',
#
#                   max_episode_steps=1000)

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
        experiences can be either dictionary (with the keys 'state', 'action', 'reward', 'is_goal' and 'next_state') or list/tuple/set the size of 3
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
        if len(experiences) != 5:
            raise ValueError(f'Each experience must be size 4: (state, action, reward, is_goal, next_state) quartet but x is {experiences}')

        if isinstance(experiences, dict):
            to_append = [experiences['state'], experiences['action'], experiences['reward'], experiences['is_goal'], experiences['next_state']]
        else:
            to_append = [experiences[0], experiences[1], experiences[2], experiences[3], experiences[4]]

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


class NeuralStateEvaluator:
    def __init__(self, **kwargs):
        self.network_layer_structure = kwargs.get('layers_structure', (100, 50, 20))
        self.dropout_rate = kwargs.get('dropout_rate', 0.5)
        self.layers_activation_func = kwargs.get('activation_function', tf.nn.relu)

        self.network_optimizer = kwargs.get('network_optimizer', tf.optimizers.RMSprop)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.network_optimizer = self.network_optimizer(lr=self.learning_rate)

        self.network_loss_function = kwargs.get('network_loss', 'mse')

        self.action_space_size = kwargs.get('action_space_size', ACTION_SPACE_SIZE)
        self.state_space_size = kwargs.get('state_space_size', STATE_SPACE_SIZE)

        # add input layer
        layers = []
        input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_space_size + self.action_space_size))
        # adding all layers
        layers.append(input_layer)
        for layer_idx, layer_size in enumerate(self.network_layer_structure):
            if isinstance(layer_size, str) == 'dropout':
                layers.append(tf.keras.layers.Dropout(self.dropout_rate, name=f'layer_{layer_idx}_dropout'))
                continue
            layers.append(
                tf.keras.layers.Dense(layer_size, activation=self.layers_activation_func, name=f'layer_{layer_idx}'))

        # adding output layer
        layers.append(tf.keras.layers.Dense(1, activation='relu'))

        # creating the model
        self.model = tf.keras.Sequential(layers)

        self.model.compile(self.network_optimizer, loss=self.network_loss_function)

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.fit(x, y, verbose=0)



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


class CustomMetric(tf.keras.metrics.Metric):
    """
    custom metric to keep track of loss in each training step
    """
    def __init__(self, name='training_step_loss', type=tf.int32):
        super(CustomMetric, self).__init__(name=name, dtype=type)
        self.val = None

    def update_state(self, x):
        self.val = x

    def result(self):
        return self.val


class ImprovedDeepQLearner(BaseQlearningAgent):
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

        super(ImprovedDeepQLearner, self).__init__(enviorment,goal_state,num_episods,max_steps_per_episode,learning_rate,learning_rate_decay,discount_rate,expolaration_decay_rate, min_expolaration_rate)
        self.exp_replay_size = kwargs.get('max_experience_replay_size', 5000)
        self.experience_replay_queue = ExperienceReplayDeque(max_deque_size=self.exp_replay_size)
        self.state_space_size = self.enviorment.observation_space.shape[0]
        self.action_space_size = self.enviorment.action_space.n
        kwargs['action_space_size'] = self.action_space_size
        kwargs['state_space_size'] = self.state_space_size
        self.enviorment._max_episode_steps = 500

        state_action_evaluator_kwargs = kwargs.get('state_action_evaluator_kwargs', {})
        self.state_action_evaluation_nn = NeuralStateEvaluator(**state_action_evaluator_kwargs)

        self.nn_target = NeuralQEstimator(**kwargs)
        self.nn_q_value = NeuralQEstimator(**kwargs)

        self.pre_train_num_of_steps = kwargs.get('pre_train_num_of_steps', 100)

        self.number_of_steps_to_update_weights = kwargs.get('number_of_steps_to_update_weights', kwargs.get('c', 10))
        self.experience_replay_sample_size = kwargs.get('experience_replay_sample_size', 50)
        # overriding the baseQlearning exploration rate attribute
        self.expolration_rate = kwargs.get('initial_exploration_rate', 0.5)

        self.is_model_trained = False

        # for tensorboard
        self.train_summary_writer = tf.summary.create_file_writer(kwargs.get('tensor_board_train_path',
                                                                             ''))
        self.test_summary_writer = tf.summary.create_file_writer(kwargs.get('tensor_board_test_path',
                                                                            ''))
        # metrics
        self.train_episode_mean_reward_metric = tf.keras.metrics.Mean('train_reward', dtype=tf.float32)
        # self.train_steps_num_metric = tf.keras.metrics.Mean(name='train_steps_num', dtype=tf.float32)
        self.train_steps_loss_metric = CustomMetric(type=tf.float32)

        self.state_action_eval_loss_metric = CustomMetric(name='state_action_eval_loss_metric', type=tf.float32)
        self.train_episode_reward_metric = CustomMetric(name='episode_reward', type=tf.int32)
        self.train_episode_100_last_reward_metric = tf.keras.metrics.Mean(name='episode_reward_100_last')



        # rendering flags
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

        is_done = False
        global_step_num = 0

        for episode_num in range(self.num_episods):
            state = self.enviorment.reset()
            is_done = False
            #     statistics
            rewards_for_q_eval_updates = []
            accumulated_reward = 0
            last_100_episodes_rewards = []
            episode_step_num = 0
            while episode_step_num < self.max_steps_per_episode and not is_done:
                # todo: remove rendering before submission:
                if self.render_during_training:
                    self.enviorment.render()
                ##################
                if global_step_num < self.pre_train_num_of_steps:
                    print(f"{'#' * 20}\nNow training state-action evaluator\n{'#' * 20}")
                    chosen_action = self.enviorment.action_space.sample()

                    if global_step_num > self.experience_replay_sample_size:
                        experiences = self.experience_replay_queue.pop_samples(self.experience_replay_sample_size)
                        samples_y = experiences[:, 2].astype('float')
                        samples_states = np.stack(experiences[:, 0])

                        samples_actions = experiences[:, 1]
                        samples_chosen_actions = np.zeros(shape=(len(samples_y), self.action_space_size))
                        # samples_rewards = experiences[:, 2]
                        samples_chosen_actions[np.arange(0, len(samples_chosen_actions), 1), samples_actions.astype(int)] = samples_y

                        samples_x = np.hstack([samples_states, samples_chosen_actions]).astype('float')
                        training_loss = self.state_action_evaluation_nn.fit(samples_x, samples_y)

                        self.state_action_eval_loss_metric(training_loss.history['loss'][0])
                else:
                    chosen_action = self.sample_action(state.reshape(1, -1))

                new_state, reward, is_done, info = self.enviorment.step(chosen_action)

                accumulated_reward += reward
                # adding to experience replay
                self.experience_replay_queue.append((state, chosen_action, reward, is_done, new_state))

                if is_done or episode_step_num == self.max_steps_per_episode - 1:
                    # self.train_steps_num_metric(episode_step_num)
                    self.train_episode_reward_metric(accumulated_reward)
                    self.train_episode_mean_reward_metric(accumulated_reward)
                    rewards_for_q_eval_updates.append(accumulated_reward)
                    global_step_num += 1

                    print('LOST!')
                    break
                if global_step_num >= self.pre_train_num_of_steps:
                    mini_batch_from_experience_replay_raw = self.experience_replay_queue.pop_samples(
                        self.experience_replay_sample_size)
                    current_q = self.nn_q_value.predict(np.stack(mini_batch_from_experience_replay_raw[:, 0]))
                    target_q = np.copy(current_q)
                    next_q = self.nn_target.predict(np.stack(mini_batch_from_experience_replay_raw[:, 4]))
                    max_next_q = np.amax(next_q, axis=1)
                    for i in range(mini_batch_from_experience_replay_raw[:, 0].shape[0]):
                        target_q[i][mini_batch_from_experience_replay_raw[i, 1]] = \
                            mini_batch_from_experience_replay_raw[i, 2] if mini_batch_from_experience_replay_raw[i, 3] else \
                                mini_batch_from_experience_replay_raw[i, 2] + self.discount_rate * max_next_q[i]

                    step_loss = self.nn_q_value.fit(np.stack(mini_batch_from_experience_replay_raw[:, 0]), target_q,
                                                    callbacks=[tf.keras.callbacks.LearningRateScheduler(
                                                        self.lr_decay_scheduler_wrapper(self.learning_rate_decay))],
                                                    )

                    self.train_steps_loss_metric(step_loss.history['loss'][0])
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss_at_training_step', self.train_steps_loss_metric.result(), step=global_step_num)

                    if episode_step_num % self.number_of_steps_to_update_weights == 0:
                        self.update_q()
                    state = new_state

                    episode_step_num += 1

                global_step_num += 1

            self.train_episode_reward_metric(accumulated_reward)

            ## Metrics
            rewards_per_episode.append(accumulated_reward)

            if self.train_episode_mean_reward_metric.result() > 475:
                consecutive_episodes_with_high_reward += 1
            else:
                consecutive_episodes_with_high_reward = 0

            if consecutive_episodes_with_high_reward == 100:
                with open('simple_stats_file_IMPROVED_DQN.txt', 'a+') as f:
                    f.write(f'{datetime.date}|{datetime.time}: '
                            f'Number of episodes until 100 consecutive '
                            f'episode with avg reward higher than 475 is {episode_num}')

            if len(last_100_episodes_rewards) >= 100:
                last_100_episodes_rewards.pop(0)

            last_100_episodes_rewards.append(accumulated_reward)

            self.train_episode_100_last_reward_metric(np.mean(np.array(last_100_episodes_rewards)))

            with self.train_summary_writer.as_default():
                tf.summary.scalar('episode_reward', self.train_episode_reward_metric.result(), step=episode_num)
                tf.summary.scalar('avg_reward', self.train_episode_mean_reward_metric.result(), step=episode_num)
                # tf.summary.scalar('number_of_steps_to_reward', self.train_steps_num_metric.result(), step=episode_num)
                tf.summary.scalar('episode_reward_100_last', self.train_episode_100_last_reward_metric.result(), step=episode_num)
                # tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

            template = '#Episode: {}, Reward: {}, Accumulated Average Reward: {}, Average 100 Last Rewards: {}'
            print(template.format(episode_num + 1,
                                  self.train_episode_reward_metric.result(),
                                  self.train_episode_mean_reward_metric.result(),
                                  self.train_episode_100_last_reward_metric.result()))


            counted_step = episode_step_num
            if (episode_step_num < self.max_steps_per_episode and not is_done):
                counted_step = self.max_steps_per_episode

            ##accumulate step per 100 episodes
            steps_per_100_episodes[episode_num // 100] += counted_step / 100

            self.expolration_rate = max((self.min_expolaration_rate, self.expolration_rate * np.exp(
                -self.expolaration_decay_rate * episode_num)))
            self.learning_rate = self.learning_rate * self.learning_rate_decay

            if episode_num > 400 and self.train_episode_mean_reward_metric.result() < 50:
                break

            if episode_num > 1000 and self.train_episode_mean_reward_metric.result() < 200: # todo: replace to train_episode_100_last_reward_metric
                break


        # todo: we probably need to append here the networks' statuses.
        # journy_q_tables.append(self.q_table)

        self.train_episode_mean_reward_metric.reset_states()
        self.train_steps_loss_metric.reset_states()
        self.train_episode_reward_metric.reset_states()
        self.train_episode_100_last_reward_metric.reset_states()

        steps_per_100_episodes = steps_per_100_episodes / 100

        self.train_summary(journy_q_tables, rewards_per_episode, steps_per_100_episodes)

        self.is_model_trained = True

    def sample_action(self, state, training=True):
        assert state.shape[1] == self.state_space_size, f'input 2nd dimension must be {self.state_space_size}'

        epsilon = random.uniform(0, 1)
        if epsilon > self.expolration_rate or not training:
            state_action_eval_scores = np.zeros(self.action_space_size)
            for action_idx in range(self.action_space_size):
                actions = np.zeros(self.action_space_size)
                actions[action_idx] = 1
                state_action_eval_scores[action_idx] = self.state_action_evaluation_nn.predict(np.hstack([state, actions.reshape(1, -1)]))

            # action = np.argmax(self.nn_q_value.predict(state)*state_action_eval_scores)
            action = np.argmax(self.nn_q_value.predict(state))
        else:
            action = self.enviorment.action_space.sample()

        return action

    def update_q(self, state=None, action=None, reward=None, new_state=None):
        self.nn_target.model.set_weights(self.nn_q_value.model.get_weights())

    def train_summary(self, journy_q_tables, rewards_per_episode, steps_per_100_episodes):
        # plot_reward_per_episode(rewards_per_episode)
        # plot_average_num_of_steps_to_reach_goal(steps_per_100_episodes)
        print('FINISHED')
        return

    def move(self, state, training=True):
        action = self.sample_action(state, training=training)
        return self.environment.step(action)

    def play(self):
        # todo: remove before submission
        import playsound
        from time import sleep, time
        ###########
        assert self.is_model_trained, f'Model must be trained first!'
        state = self.environment.reset()

        # todo: remove before submission
        print(f"{'#'*10}\nSTARTED PLAYING\n{'#'*10}")
        # wait for 3 seconds
        sleep(3)
        ###########

        step = 1
        if self.render_during_testing:
            self.environment.render()
        state, reward, is_goal, info = self.move(state.reshape(1, -1), training=False)
        seconds_remaining_stable = time()
        while (not is_goal):
            print("Step # {}:".format(step))
            self.environment.render()
            state, reward, is_goal, info = self.move(state.reshape(1, -1), training=False)
            step += 1
        seconds_remaining_stable = time() - seconds_remaining_stable
        # todo: figure out if there is a goal state
        if (is_goal):
            print("LOSE! # of seconds remaining stable is:{}".format(seconds_remaining_stable))
        else:
            print("NO LOSE! # of seconds is:{}".format(seconds_remaining_stable))
        return None


nn_q_learner = None


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


def run_hyper_params_search(params_tuples_to_run: List[Tuple] = None):
    global nn_q_learner

    if params_tuples_to_run == None:
        lr_options = np.arange(0.1, 0.21, 0.1)  # 2
        lr_d_options = np.array([0.9 + x for x in [0.099999999999, 0.099]])  # 1
        discount_options = np.array([0.9 + x for x in (0, 0.09)])  # 2
        n_steps_to_update_options = np.arange(20, 101, 80)  # 2
        network_struct_options = ((10, 10, 10), (10, 10, 10, 10, 10)) # 2
        optimizers_options = (tf.optimizers.RMSprop, tf.optimizers.Adam) # 2
        params_tuples_to_run = list(itertools.product(lr_options, lr_d_options, discount_options, n_steps_to_update_options, network_struct_options, optimizers_options))

    exp_idx = 0
    total_exps = len(params_tuples_to_run)
    for lr, lr_d, discount, n_steps_to_update, net_structure, net_optimizer in params_tuples_to_run:
        print(
            f'running experiment: {exp_idx + 1}/{total_exps}:\nlr:{lr}|lr_d:{lr_d}|discount:{discount}|n_update:{n_steps_to_update}'
            f'|structure:{net_structure}|optimizer:{net_optimizer}')
        exp_idx += 1

        regex = re.compile(r'\W*')
        q_estimator_kwargs = {
            'lr': lr,
            'lr_d': lr_d,
            'discount': discount,
            'n_steps_to_update': n_steps_to_update,
            'net_structure': net_structure,
            'net_optimizer': net_optimizer
        }

        kwargs_name = '_'.join([f'{regex.sub("", str(key))}={regex.sub("", str(val))}' if not type(val) == type(
            tf.optimizers.Optimizer) else str(val)[32:-2] for key, val in
                                q_estimator_kwargs.items()])[:]

        # print(kwargs_name)
        # param configuration:
        learning_rate = lr
        learning_rate_decay = lr_d
        discount_rate = discount
        expolaration_decay_rate = 0.001
        initial_exploration_rate = 0.5
        min_expolaration_rate = 0.005
        layers_structure = net_structure
        network_optimizer = net_optimizer
        epsilon_greedy = 0.1
        num_episods = 2000
        max_steps_per_episode = 500
        number_of_steps_to_update_weights = n_steps_to_update

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


        DIR_FOR_TF_BOARD_TRAIN_LOGS = os.sep.join([os.getcwd(), f'Section3Results', f'tb_callback_dir', kwargs_name, 'train'])
        DIR_FOR_TF_BOARD_TEST_LOGS = os.sep.join([os.getcwd(), f'Section3Results', f'tb_callback_dir', kwargs_name, 'test'])

        q_estimator_kwargs.update(
            {
                'tensor_board_train_path': DIR_FOR_TF_BOARD_TRAIN_LOGS,
                'tensor_board_test_path': DIR_FOR_TF_BOARD_TEST_LOGS,
                'state_action_evaluator_kwargs': {
                    'layers_structure': (10, 5),
                    'learning_rate': 0.1,
                    'network_loss': 'mse',
                }
            }
        )

        if os.path.isdir(DIR_FOR_TF_BOARD_TRAIN_LOGS):
            shutil.rmtree(DIR_FOR_TF_BOARD_TRAIN_LOGS, ignore_errors=True)
        if os.path.isdir(DIR_FOR_TF_BOARD_TEST_LOGS):
            shutil.rmtree(DIR_FOR_TF_BOARD_TEST_LOGS, ignore_errors=True)

        # initialization:
        nn_q_learner = ImprovedDeepQLearner(
            enviorment=ENVIRONMENT,
            goal_state=None,
            render_training_flag=False,
            render_testing_flag=True,
            **q_estimator_kwargs
        )

        train_agent()

if __name__ == '__main__':
    run_hyper_params_search()

