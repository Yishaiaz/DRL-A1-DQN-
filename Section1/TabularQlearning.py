import numpy as np
import random
import gym
import q_value_table_color_map,plot_reward_per_episode from section1_visualization

class TabularQlearningAgent:
    
    def __init__(self,enviorment,goal_state,num_episods=5000,max_steps_per_episode=100,learning_rate=0.01,discount_rate=0.99,expolaration_decay_rate=0.001, min_expolaration_rate=0.01) -> None:
        
        ## Initiate enviorment 
        self.enviorment = enviorment

        ## Initiate training parameters
        self.num_episods = num_episods
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.expolaration_decay_rate = expolaration_decay_rate
        self.min_expolaration_rate = min_expolaration_rate
        self.expolration_rate = 1
        
        ## Initiate agent data members
        action_space_size = enviorment.action_space.n
        state_space_size = enviorment.observation_space.n
        self.goal_state = goal_state
        self.q_table = np.zeros((state_space_size,action_space_size))
        # self.rewards = []
        
    def train(self):
        journy_q_tables = []
        rewards_per_episode = []
        steps_per_100_episodes = []
        for episode in range(self.num_episods):
            state = self.enviorment.reset()
            episode_rewards = 0
            step= 0
            is_done = False
            is_goal = False
            while (step < self.max_steps_per_episode and not is_done):
                    
                action = self.sample_action(state)
                
                new_state,reward,is_done,info = self.enviorment.step(action)
                if (new_state == self.goal_state):
                    is_goal = True
                self.update_q(state,action,reward,new_state)
                state = new_state
                episode_rewards += reward
            
            ## Metrics
            ## Reward per episode  
            rewards_per_episode.append(episode_rewards)
            if (episode % 100 == 0):
                steps_per_100_episodes.append(0)   
            
            counted_step = step
            if (step < self.max_steps_per_episode and not is_goal):
                counted_step = self.max_steps_per_episode
            
            ##Avergae step per 100 episodes
            steps_per_100_episodes[episode // 100]+=counted_step/100
            
            ##Q table at 500,2000 and at the end
            if (episode == 500 or episode == 2000):
                journy_q_tables.append[self.q_table]   
            self.expolration_rate = self.min_expolaration_rate + (1-self.min_expolaration_rate)*np.exp(-self.expolaration_decay_rate*episode)
        
        journy_q_tables.append[self.q_table]   
        self.train_summary(journy_q_tables, rewards_per_episode,steps_per_100_episodes)                      
     
    def sample_action(self,state):
        epsilon = random.uniform(0,1)
        if epsilon > self.expolration_rate: 
            action = np.argmax(self.get_q_options(state))
        else:
            action = self.enviorment.action_space.sample()

        return action
        
    def update_q(self,state,action,reward,new_state):
        self.q_table[state,action] = self.q_table[state,action] * (1-self.learning_rate) + self.learning_rate * (reward + self.discount_rate * np.argmax(self.q_table[new_state,:]))
    
    def get_q_options(self,state):
        return self.q_table[state,:]
    
    def sample_action_test(self,state):
        action = np.argmax(self.get_q_options(state))
        return action
    
    def train_summary(self,journy_q_tables, rewards_per_episode,steps_per_100_episodes):
        

if __name__ == '__main__':
    # briefly testing TabularQlearning
    enviorment = gym.make("FrozenLake-v0")
    agent = TabularQlearningAgent(enviorment)
    agent.train()
    