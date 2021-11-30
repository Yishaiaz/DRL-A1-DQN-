import numpy as np
import gym
from visualization import q_value_table_color_map, plot_reward_per_episode, plot_average_num_of_steps_to_reach_goal
from BaseQlearning import BaseQlearningAgent


class TabularQlearningAgent(BaseQlearningAgent):

    def __init__(self, enviorment, goal_state, num_episods=5000, max_steps_per_episode=100, learning_rate=0.01, learning_rate_decay=0.99, discount_rate=0.99, expolaration_decay_rate=0.001, min_expolaration_rate=0.01, expolration_rate=1) -> None:

        super(TabularQlearningAgent, self).__init__(enviorment, num_episods, max_steps_per_episode, learning_rate,
                                                    learning_rate_decay, discount_rate, expolaration_decay_rate, min_expolaration_rate, expolration_rate)

        # Initiate agent data members
        self.state_space_size = self.enviorment.env.observation_space.n
        self.action_space_size = self.enviorment.env.action_space.n
        self.goal_state = goal_state
        self.q_table = np.zeros(
            (self.state_space_size, self.action_space_size))

    def update_q(self, state, action, reward, new_state):
        self.q_table[state, action] = self.q_table[state, action] * (1-self.learning_rate) + self.learning_rate * (
            reward + self.discount_rate * np.max(self.q_table[new_state, :]))

    def get_q_options(self, state):
        return self.q_table[state, :]

    def train_summary(self, journy_q_tables, rewards_per_episode, steps_per_100_episodes):
        plot_reward_per_episode(rewards_per_episode, plot_type='bar')
        plot_average_num_of_steps_to_reach_goal(steps_per_100_episodes)
        q_value_table_color_map(journy_q_tables[0], 500, np.arange(
            self.state_space_size), ['LEFT', 'DOWN', 'RIGHT', 'UP'], reverse_axis=True)
        q_value_table_color_map(journy_q_tables[1], 2000, np.arange(
            self.state_space_size), ['LEFT', 'DOWN', 'RIGHT', 'UP'], reverse_axis=True)
        q_value_table_color_map(journy_q_tables[2], 5000, np.arange(
            self.state_space_size), ['LEFT', 'DOWN', 'RIGHT', 'UP'], reverse_axis=True)


if __name__ == '__main__':
    enviorment = gym.make("FrozenLake-v1", is_slippery=True)
    agent = TabularQlearningAgent(enviorment, 15, num_episods=5000, max_steps_per_episode=100, learning_rate=0.2, learning_rate_decay=0.999,
                                  discount_rate=0.99, expolaration_decay_rate=0.001, min_expolaration_rate=0.01, expolration_rate=0.1)
    agent.train()
    agent.play()
