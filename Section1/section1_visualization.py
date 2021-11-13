import os
from typing import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def q_value_table_color_map(q_table_values: np.array,
                            n_steps: int,
                            states: np.array,
                            actions: np.array, **kwargs) -> None:
    """
    this function assumes the following structure:
    values_table.shape[0] == len(states) and values_table.shape[1] == len(actions).
    the plotted heatmap will plot the states as the x-axis and the actions as the y-axis.
    to reverse the order, pass the keyword argument reverse_axis=True
    :param q_table_values: 2D np array
    :param n_steps: int
    :param states: np.array
    :param actions: np.array
    :return:
    """
    plt.clf()

    fig, ax = plt.subplots()
    im = ax.imshow(q_table_values)

    if kwargs.get('reverse_axis', False):
        ax.set_yticks(np.arange(q_table_values.shape[0]))
        ax.set_xticks(np.arange(q_table_values.shape[1]))
        ax.set_yticklabels(states)
        ax.set_xticklabels(actions)
        x_axis_len = len(actions)
        y_axis_len = len(states)
    else:
        ax.set_xticks(np.arange(q_table_values.shape[0]))
        ax.set_yticks(np.arange(q_table_values.shape[1]))
        ax.set_xticklabels(states)
        ax.set_yticklabels(actions)
        x_axis_len = len(states)
        y_axis_len = len(actions)

    for i in range(x_axis_len):
        for j in range(y_axis_len):
            text = ax.text(j, i, q_table_values[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(f"Q values heatmap after {n_steps} steps")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.tight_layout()
    plt.show()

    path_to_save_fig = os.sep.join([os.getcwd(),'\\Results\\Section1',
                                    'q_value_table_color_map'])
    plt.savefig(path_to_save_fig + '.png', dpi=200)
    plt.savefig(path_to_save_fig + '.eps', dpi=200)

    plt.close(fig)


def plot_reward_per_episode(reward_per_episode: np.array, **kwargs) -> None:
    """
    plots the reward of the value at each episode as a bar plot.
    if you wish to use scatter/plot pass the keyword argument plot_type='scatter' or plot_type='plot'
    :param reward_per_episode: the value of the reward per episode
    :param kwargs:
    :return:
    """
    plt.clf()

    fig, ax = plt.subplots()
    x_axis = np.arange(1, len(reward_per_episode)+1, 1)
    episodes_amount = x_axis.max()

    if kwargs.get('plot_type', 'bar'):
        ax.bar(x_axis, reward_per_episode)

    elif kwargs.get('plot_type', 'scatter'):
        ax.scatter(x_axis, reward_per_episode)

    elif kwargs.get('plot_type', 'plot'):
        ax.plot(x_axis, reward_per_episode)

    ax.set_xticks(x_axis)
    ax.set_xticklabels([f'{e_num}' for e_num in x_axis])
    ax.set_title(f'Reward Per Episode\n{episodes_amount}# of episodes')
    ax.set_xlabel(f'Episode number')
    ax.set_ylabel(f'Reward')

    path_to_save_fig = os.sep.join(
        [os.getcwd(),'Results\\Section1',
         f'reward_per_episode_{episodes_amount}_episodes'])

    plt.savefig(path_to_save_fig + '.png', dpi=200)
    plt.savefig(path_to_save_fig + '.eps', dpi=200)

    plt.close(fig)


def plot_average_num_of_steps_to_reach_goal(steps_per_episode):
# todo:     Plot of the average number of steps to the goal over last 100 episodes (plot
#  every 100 episodes). If agent didn't get to the goal, the number of steps of
#  the episode will be set to 100.
    pass