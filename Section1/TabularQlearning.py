"""-Create a lookup table containing the approximation of the Q-value for each
state-action pair.
- Initialize the table with zeros.
- Choose initial values for the hyper-parameters: learning rate 𝛼, discount
factor 𝛾, decay rate for decaying epsilon-greedy probability.
- Follow the algorithm presented in figure 3. This is exactly the same
algorithm from figure 2, written slightly differently.

-You should sample actions using decaying 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦 or some other
method of your choice.
- Optimize the hyper-parameters for optimal result
- The Agent should train over 5000 episodes with a maximum of 100 steps
in an episode.
"""

# todo: Lion in charge of implementing:
# required interface:
#     1 - init input: action space size, reward
    2 - train
    3 -

OPENAI gym:
https://gym.openai.com/envs/FrozenLake-v0/
states:
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)