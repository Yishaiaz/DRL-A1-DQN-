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