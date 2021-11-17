import numpy as np
import random
import gym
from visualization import q_value_table_color_map,plot_reward_per_episode,plot_average_num_of_steps_to_reach_goal
from abc import ABC

class BaseQlearningAgent(ABC):
    
    def __init__(self,enviorment,goal_state,num_episods,max_steps_per_episode,learning_rate,learning_rate_decay,discount_rate,expolaration_decay_rate, min_expolaration_rate) -> None:
        
        ## Initiate enviorment 
        self.enviorment = enviorment

        ## Initiate training parameters
        self.num_episods = num_episods
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.discount_rate = discount_rate
        self.expolaration_decay_rate = expolaration_decay_rate
        self.min_expolaration_rate = min_expolaration_rate
        self.expolration_rate = 1
        
        ## Initiate agent data members
        self.action_space_size = enviorment.action_space.n
        self.state_space_size = enviorment.observation_space.n
        self.goal_state = goal_state
        
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
                step+=1
            
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
                journy_q_tables.append(self.q_table.copy())
            self.expolration_rate = self.min_expolaration_rate + (1-self.min_expolaration_rate)*np.exp(-self.expolaration_decay_rate*episode)
            self.learning_rate = self.learning_rate * self.learning_rate_decay
        
        journy_q_tables.append(self.q_table)
        self.train_summary(journy_q_tables, rewards_per_episode,steps_per_100_episodes)                      
     
    def sample_action(self,state):
        epsilon = random.uniform(0,1)
        if epsilon > self.expolration_rate: 
            action = np.argmax(self.get_q_options(state),)
        else:
            action = self.enviorment.action_space.sample()

        return action
        
    def update_q(self,state,action,reward,new_state):
        pass ## abstract method
    
    def get_q_options(self,state):
        pass ## abstract method
    
    def move(self,state):
        action = np.argmax(self.get_q_options(state))
        return self.enviorment.step(action)
    
    def train_summary(self,journy_q_tables, rewards_per_episode,steps_per_100_episodes):
        # plot_reward_per_episode(rewards_per_episode)
        # plot_average_num_of_steps_to_reach_goal(steps_per_100_episodes)
        # q_value_table_color_map(journy_q_tables[0],500,np.arange(15),['LEFT','DOWN','RIGHT','UP'])
        return
    
    def play(self):
        state = self.enviorment.reset()
        step = 1
        self.enviorment.render()
        state,reward,is_done,info = self.move(state)
        while (not is_done):
            print("Step # {}:".format(step))
            self.enviorment.render()
            state,reward,is_done,info = self.move(state)    
            step+=1
        
        if (state == self.goal_state):
            print ("Win! # of steps is:{}".format(step))
        else:
            print ("Lose! # of steps is:{}".format(step))
        return None

    