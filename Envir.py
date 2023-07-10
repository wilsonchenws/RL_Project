# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 23:01:26 2023

@author: wilsonchenws
"""

# This is a practice script to better understand reinforcement learning.
# My goal is to train a model to play a simple game in which agent need to find optimal.
# route and sequence to retrieve treasures.

# Then visualize the result with Pygame???? 
# Change the experience into a structured numpy
# Change experiences into a numpy storing structured numpy


import numpy as np
import math




#%% Class Design
def move(action):
    
    list_action = ['d', 'u', 'r', 'l']
    output = [1 if action == i else 0 for i in list_action]
    
    return output
    
class envir(object):
    
    # Need to specify the size, fuel and prize num
    def __init__(self, size_of_map, fuel, gamma):
        
        # Parameter
        self.size_of_map = size_of_map
        self.total_fuel = np.int64(fuel)
        self.gamma = gamma
        self.world_map = np.zeros(size_of_map*size_of_map)
        
        # Initializing the where the prize
        self.loc_player = np.random.choice(self.size_of_map * self.size_of_map, 1)
        self.pos_player =  self.loc_to_pos(self.loc_player)
        
        self.loc_prize = np.random.choice(self.size_of_map * self.size_of_map, 10, replace = False)
        self.value_of_prize = np.random.choice(np.arange(1,11), 
                                               10,
                                               replace = True)
        
        self.pos_prize = self.loc_to_prize_loc(self.loc_prize, self.value_of_prize)

        
        
        
        #Initializing some state variables
        self.dis_to_top = self.pos_player[0] 
        self.dis_to_button = self.size_of_map - self.pos_player[0] -1
        self.dis_to_left = self.pos_player[1]
        self.dis_to_right = self.size_of_map - self.pos_player[1] - 1
        
        
        # This is to record how may steps have been pased
        self.total_step = 0
        self.total_return = 0
        self.current_fuel = fuel
        
        # This is to record historical record
        self.history_score = np.array([])
        
        
        for loc, prize in zip(self.loc_prize, self.value_of_prize):
            self.world_map[loc] = prize
        
        
    
    def loc_to_prize_loc(self, loc_list, value_list):
        prize_loc = []
        for loc, value in zip(loc_list, value_list):
            pos_tuple = self.loc_to_pos(np.array([loc]))
            prize_loc.append(pos_tuple[0])
            prize_loc.append(pos_tuple[1])
            prize_loc.append(value)
    
        return prize_loc
    
    def pos_to_loc(self, pos):
        row, column = pos
        loc = self.size_of_map * row + column
        return loc
    
    def loc_to_pos(self, loc):
        return ([math.floor(loc[0] / self.size_of_map), (loc[0] % self.size_of_map)])
        
        
    def visualize_map(self):
        matrix_shape_map = self.world_map.reshape(self.size_of_map, self.size_of_map)
        return (matrix_shape_map)
    
    def pos_player_tuple(self):
        return

    def reset(self, game_size ,new_fuel):
        # This function would reset the whole environment.
        
        # Record the reward
        
        
        #Set new total fuel and reset fuel level
        self.total_fuel = new_fuel
        self.current_fuel = new_fuel
        # Add Score to historicacl record and reset total score
        self.history_score = np.append(self.history_score, self.total_return)
        self.total_return = 0
        
        # Reset step count
        self.total_step = 0
        
        #
        self.size_of_map = game_size
        
        # Reset location of prize
        self.loc_prize = np.random.choice(self.size_of_map * self.size_of_map, 10, replace = False)
        
        self.value_of_prize = np.random.choice(np.arange(1, 11), 
                                               10,
                                               replace = False)
        # Reset prize's position
        self.pos_prize = self.loc_to_prize_loc(self.loc_prize, self.value_of_prize)
        
        # Reset world map
        self.world_map = np.zeros(self.size_of_map * self.size_of_map)
        
        
        for loc, value in zip(self.loc_prize, self.value_of_prize):
            self.world_map[loc] = value
        
        self.loc_player = np.random.choice(self.size_of_map * self.size_of_map, 1)
        
        self.pos_player = self.loc_to_pos(self.loc_player)
        
        #Initializing some state variables
        self.dis_to_top = self.pos_player[0] 
        self.dis_to_button = self.size_of_map - self.pos_player[0] -1
        self.dis_to_left = self.pos_player[1]
        self.dis_to_right = self.size_of_map - self.pos_player[1] - 1
        
        
        
        return None
    
    def returnCurrentState(self):
        
        position_of_player = np.array([self.dis_to_top, self.dis_to_button, self.dis_to_left, self.dis_to_right])
        
        current_state = np.concatenate([position_of_player, self.pos_player, self.pos_prize, self.current_fuel])
        
        return current_state.reshape(1, 37)/self.size_of_map
        
        
    def step(self, action, epsilon):
        

        # 先立刻回傳移動前的狀況
        current_state = self.returnCurrentState()
        # 計算當下的Reward(玩家所出的環境可以獲得的報酬)        
        reward, done = self.calculate_reward()
        
        #計算總分(有Gamma, Discount Factor) 
        self.total_return += reward * (self.gamma ** self.total_step)
        
        #Update一下分數紀錄
        if self.loc_player in self.loc_prize:
            loc_of_list = np.where(self.loc_prize == self.loc_player[0])[0][0]
            self.value_of_prize[loc_of_list] = 0
            self.pos_prize = self.loc_to_prize_loc(self.loc_prize, self.value_of_prize)
            self.world_map[self.loc_player] = 0
        
        
        maxReturnAction = np.argmax(action)
        actualAction = np.array( [1 if i == maxReturnAction else 0 for i in range(4)] ).reshape(1, 4)
        

        # Original Action
        if not done:
            
        # implementation of epsilon-greedy algorithm
            if (np.random.uniform(0,1,1) <= epsilon):
                maxReturnAction = np.random.choice(4)
                actualAction = np.zeros(4).reshape(1,4)
                actualAction[0, maxReturnAction] = 1
        
        # 0:down 1:up, 2: right, 3: left
        
        # Change the state based on our action
            if maxReturnAction == 0:
                self.pos_player[0] += 1
                self.dis_to_top += 1
                self.dis_to_button -= 1
                
                
            elif maxReturnAction == 1:
                self.pos_player[0] -= 1
                self.dis_to_top -= 1
                self.dis_to_button += 1
                
                
            elif maxReturnAction == 2:
                self.pos_player[1] += 1
                self.dis_to_right -= 1
                self.dis_to_left += 1

            else:
                self.pos_player[1] -= 1
                self.dis_to_right += 1
                self.dis_to_left -=1
        
            # Update player's location, world map and fuel level
            
            self.loc_player = np.array([self.pos_to_loc(self.pos_player)])
            
            
            
            

             
            
            
            self.current_fuel -= 1
            
            next_state = self.returnCurrentState()
        
            self.total_step += 1
        
            return np.concatenate([current_state, actualAction, reward.reshape(1, 1)/10, next_state, done.reshape(1, 1)], axis = 1)
        
        else:
            # If it's terminal state, the reward will be the total reward, next state will be same as current state
            return np.concatenate([current_state, actualAction, reward.reshape(1, 1)/10, current_state, done.reshape(1, 1)], axis = 1)
        

    def calculate_reward(self):
        
        # The function will calculate the return of current state, and evaluate whether the episode end or not.
        reward = np.array([0])
        done = np.array([0.0])
        # if player's position is out of range, then we need to punish.
        
        if (self.pos_player[0] < 0) or (self.pos_player[1] < 0):
            reward = np.array([-10])
            done = np.array([1.0])
        elif (self.pos_player[0] >= self.size_of_map) or( self.pos_player[1] >=  self.size_of_map):
            reward = np.array([-10])
            done = np.array([1.0])
        elif (self.current_fuel == 0):
            reward = self.world_map[self.loc_player]
            done = np.array([1.0])
        # If above condition is not met, this line of code should not produce error
        
        else:
            reward = self.world_map[self.loc_player]

        
        return reward, done
    
    def show(self):
        
        print(f'Current Reward is :{self.total_return}\n Player at {self.pos_player} with fuel of {self.current_fuel}\n World Map is\n {self.visualize_map()}')
        



# The loss function should works fine.




#%% Execution
if __name__ == '__main__':
    size_of_game = 10
    gamma = 0.9
    game = envir(size_of_game, np.array([20]), gamma = gamma)

    game.show()
    game.step(move('d'), 0)




