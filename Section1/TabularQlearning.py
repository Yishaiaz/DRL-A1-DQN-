import numpy as np
import gym
from BaseQlearning import BaseQlearningAgent

class TabularQlearningAgent(BaseQlearningAgent):
    
    def __init__(self,enviorment,goal_state,num_episods=5000,max_steps_per_episode=100,learning_rate=0.01,learning_rate_decay=0.99,discount_rate=0.99,expolaration_decay_rate=0.001, min_expolaration_rate=0.01) -> None:
        
        super(TabularQlearningAgent, self).__init__(enviorment,goal_state,num_episods,max_steps_per_episode,learning_rate,learning_rate_decay,discount_rate,expolaration_decay_rate, min_expolaration_rate)
        self.q_table = np.zeros((self.state_space_size,self.action_space_size))
        
    def update_q(self,state,action,reward,new_state):
        self.q_table[state,action] = self.q_table[state,action] * (1-self.learning_rate) + self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[new_state,:]))
    
    def get_q_options(self,state):
        return self.q_table[state,:]


if __name__ == '__main__':
    # briefly testing TabularQlearning
    enviorment = gym.make("FrozenLake-v1",is_slippery = True)
    # enviorment.render()
    agent = TabularQlearningAgent(enviorment,15,num_episods=5000,max_steps_per_episode=100,learning_rate=0.5,learning_rate_decay=0.9995, discount_rate=0.99,expolaration_decay_rate=0.001, min_expolaration_rate=0.05)
    agent.train()
    agent.play()
    