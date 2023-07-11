# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 16:38:54 2023

@author: wilsonchenws
"""
#加入Priotitized Memory Replay. (因為我大部分的Data應該都是0 transition )https://zhuanlan.zhihu.com/p/38358183
#試著Rescale my data 
#%% Import necessary model
from Envir import envir, move
from tensorflow.python.keras.layers import Dense, Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Sequential
from tf_agents.utils.common import soft_variables_update
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np


#%% Priority Replay Implementation

class SumTree(object):
    '''
    Sumtree是一個樹狀結構，所有Parent Leaves用來儲存priority value加總
    實際上數值都儲存在Leaf
    '''
    
    data_pointer = 0
    
    def __init__(self, capacity):
        
        """
        Tree structure and array storage:
        Tree index:
                0         -> storing priority sum
            /     \
          1        2
         / \      /   \
        3   4    5     6    -> storing priority for transitions
       / \ /\   /\     /\
     7  8 9 10 11 12 13 14
     
     左邊的leaf node的index一定是Parent Node index * 2 +1 , 右邊會是 * 2 + 2 
     
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        
        # Memory Buffer的大小
        self.capacity = capacity # for all priority values
        
        
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity

        self.data = np.zeros(capacity, dtype = object)
        # [--------------data frame-------------]
        #             size: capacity
    
    def add(self, p, data):
        
        # 因為儲存priority的位置在Parent Node (Capacity後面，所以記得從capacity - 1 後面開始Update)
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data # update data_frame
        
        #Update Tree
        self.update(tree_idx, p)
        
        
        self.data_pointer += 1
        #當Memory Buffer用完的時候就從位置0開始新增資料
        if self.data_pointer >= self.capacity:  #Replace when exceed the capacity
            self.data_pointer = 0
    
    def update(self, tree_idx, p):
        
        # 計算出p的變化量，我們要將所有的P值Update
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        
        # then propogate the change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
            
    def get_leaf(self, v):
        
        # v是要取的值的範圍，也就是後來抽樣的時候要傳入的
        # 我們要回傳Priority, Tree-index, 以及Data內儲存的資料
        parent_idx = 0
        
        while True:
            
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            
            if cl_idx >= len(self.tree):  #reach botton, end search
                leaf_idx = parent_idx
                break
            
            else:    # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
                    
        data_idx = leaf_idx - self.capacity + 1 
        
        return data_idx, leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def total_p(self):
        return self.tree[0]

class Memory(object):
    
    epsilon_replay = 0.01
    alpha_replay = 0.6 #[0~1] convert the importance of TD error to priority
    beta = 0.4 # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    
    abs_err_upper = 1. #clipped abs error
    
    def __init__(self, capacity):
        self.STree = SumTree(capacity)
    
    def store(self, transition):
        
        #第一次看到transition時，將TD-Error設定為最大, 確保所有經驗至少被回放一次 (若還沒有weight, 設定為1)
        max_p = np.max(self.STree.tree[-self.STree.capacity:])
        
        if max_p == 0:
            max_p = self.abs_err_upper
        
        self.STree.add(max_p, transition) # set the max p for new p
    
    def sample(self, n):
        
        #只有sample到的transition才會重新計算TD-Error, 是為了避免重新計算整個經驗池的負擔
        #ISWeight是用來解決Pirority Replay引進的Bias
        
        #n 是sample size
        # b_idx是一個one-dimension, n個
        b_idx = np.empty((n,), dtype = np.int32)
        # b_idx,是一個two-dimesion,  n x data (State, Action, R, Next State)
        b_memory  = np.empty((n, self.STree.data[0].size))
        # Importance-Sampling Weights
        ISWeights = np.empty((n,1))
        
        priority_segment = self.STree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling]) #max = 1
        
        min_prob = np.min(self.STree.tree[-self.STree.capacity:]) / self.STree.total_p
        
        #開始抽樣
        for i in range(n):
            
            a, b = priority_segment * i, priority_segment * (i + 1)
            v = np.random.uniform(a, b)
            
            data_idx, idx, p , data = self.STree.get_leaf(v)
            prob = p / self.STree.total_p
            
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            
            b_idx[i], b_memory[i, :] = idx, data
        
        return b_idx, b_memory, ISWeights
    
    def batch_update(self, tree_idx, abs_errors):
        
        #epsilon是為了避免出現TD Error恰巧為0的邊緣情況，希望這些transition一樣有機會被抽取
        abs_errors += self.epsilon_replay
        
        # 如果abs_errors太低的話，會被給予1., 確保transition不會完全不會被取到
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        
        ps = np.power(clipped_errors, self.alpha_replay)
        
        for ti, p in zip(tree_idx, ps):
            self.STree.update(ti, p)

                
#%% Define Cost function and learning function
def compute_loss(transitions, ISWeights , gamma):
    
    
    
        # <----current_state----><---action_size---><--Rewards--><-----next state----------><-done vals->
        # 7 + size_of_map * 3             4              1            7 + size_of_map * 3        1
    
    
    states = transitions[:, 0 : 37]
    actions = transitions[:, 37 : 41]
    rewards = transitions[:, 41]
    next_states = transitions[:,42:-1]
    done_vals = transitions[:, -1]
    
    
    
    # This step will find the maximum value of each output
    max_qsa = tf.reduce_max(target_q_network(next_states), axis = -1)
    
    # if the state is terminate at next step, we should use rewards instead of bell equation
    y_targets = tf.cast(tf.reshape(rewards,[-1]), tf.float32) + gamma * max_qsa * tf.cast((tf.reshape(1 - done_vals, [-1])), tf.float32)
    
    
    q_values = q_network(states)
    q_values = tf.reduce_sum(tf.math.multiply(q_values, tf.cast(actions, tf.float32)), axis = -1)
    
    abs_diff = tf.math.abs(q_values - y_targets)
    
    loss = tf.reduce_mean(tf.cast(ISWeights, tf.float32) * tf.math.squared_difference(q_values, y_targets))

    return loss, abs_diff

#這邊的Learning Function先抄Coursera的，之後再學Decorator和裡面的定義
@tf.function
def agent_learn(transitions, ISWeights ,gamma):
    '''
    Update the weights of the Q networks

    Parameters
    ----------
    experiences : A dictionary contains deque for difference experiences
        These are data set we collected in the game..
    gamma : Float
        This is the discount factor for our rewards.
        When the agent get the reward later in the game, it should receive less rewards.
        This will help agent to learn postpone punishment(Negative reward) and gain
        large reward eariler.

    Returns
    -------
    This functin is used to update Q-Target Network and Q network.

    '''
    # Sampling data
    with tf.GradientTape() as tape:
        loss, abs_diff = compute_loss(transitions, ISWeights, gamma)
    
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    
    # Soft update the Q target network
    soft_variables_update(q_network.trainable_variables, target_q_network.trainable_variables, tau = 0.1)
    
    return abs_diff

def softmax_action_sampling(q_values, temperature_factor):
    
    
    total_z = np.exp(q_values / temperature_factor).sum()
    probability = np.exp(q_values / temperature_factor) / total_z
    Action_Tree = SumTree(capacity = 4)
    for p in probability:
        Action_Tree.add(p, p)
    
    data_idx, idx, p , data = Action_Tree.get_leaf(np.random.uniform(0, 1))
    
    action = np.array( [1 if i == data_idx else 0 for i in range(4)]).reshape(1, 4)
    
    
    return action
    


#%% Parameter & Environment Setup

game_size = 10
num_of_action = 4
state_size = 37
# Distance to 4 direction, 2 input for player position and 30 input for prize distance and 1 input for remaining fuel


#設定訓練參數
alpha = 0.0001#Learning Rate
memory_size = 10000
replay_period = 4
num_episode = 100000 #環境重新設定的次數, 因為我一定有terminal state, 因此不設定強制結束條件

# Epsilon-Greedy Policy
start_epsilon = 1 # epsilon-greedy 參數, 會逐漸下調到0.05
interim_epsilon = 0.3
end_epsilon = 0.05



# Neural Network Related Parameter
explore_period = 0.5
mini_batch_size = 128
gamma = 0.9
update_counter = 0
# memory buffer是一個#memory buffer只會記得最後10萬筆


# Softmax Exploration Policy
start_temperature = 0.1
end_temperature = 0.01
decrement_temperature = (start_temperature - end_temperature) / (num_episode * explore_period)

memory_buffer = Memory(memory_size)



#%% Dueling Double DQN
inputs = Input(shape = state_size)
shared_layer = Dense(256, activation = 'relu')(inputs)
shared_layer = Dense(128, activation = 'relu')(shared_layer)
shared_layer = Dense(64, activation = 'relu')(shared_layer)

val_stream = Dense(32, activation = 'relu')(shared_layer)
val_stream = Dense(16, activation = 'relu')(val_stream)
val_stream = Dense(1)(val_stream)
adv_stream = Dense(32, activation = 'relu')(shared_layer)
adv_stream = Dense(16, activation = 'relu')(adv_stream)
adv_stream = Dense(num_of_action)(adv_stream)

q_vals = Lambda(lambda x: x[0] + x[1] - tf.reduce_mean(x[1], axis = 1, keepdims = True))([val_stream, adv_stream])


q_network = Model(inputs = inputs, outputs = q_vals)
target_q_network = Model(inputs = inputs, outputs = q_vals)
target_q_network.set_weights(q_network.get_weights())


optimizer = Adam(learning_rate = alpha)

#%% Model Training Process

current_temperature = start_temperature

gameEnvir = envir(10, np.array([20]), gamma = gamma)


for i in range(num_episode):
    if (i % 5000 == 0):
        print(f'Training Progress: {i/1000 }%')
        target_q_network.set_weights(q_network.get_weights())
    # Whever the environment restart
    
    new_fuel = np.array([20])
    
    gameEnvir.reset(game_size, new_fuel)
    
    
    # 這邊將Epsilon下降的速度調快一些，在50000個Epsiode之後就降為end_epsilon
    if i <= (num_episode * explore_period):

        current_epsilon = (start_epsilon - end_epsilon)* (num_episode * explore_period - i)  / (num_episode * explore_period)
        
        current_temperature -= decrement_temperature
        current_temperature = max(end_temperature, current_temperature)
    
    else:
        current_epsilon = end_epsilon
    #print(current_epsilon)
    #開始循環
    done = False
    
    
    while not done:
        
        current_state = gameEnvir.returnCurrentState()
        
        qvalue = q_network(current_state)
        qvalue = tf.reshape(qvalue, [-1])
        
        update_counter += 1
        
        
        #softmax exploration policy
        #soft_max_q = softmax_action_sampling(qvalue, temperature_factor = 0.01)
        
        
        transition = gameEnvir.step(qvalue, current_epsilon)
        memory_buffer.store(transition)
        done = transition[0, -1]
        
        # <----current_state----><---action_size---><--Rewards--><-----next state---------->
        # 7 + size_of_map * 3             4              1            7 + size_of_map * 3
        
        
    
        
        # Check update status:
        # 必須要大於memory_size，才可以開始學習
        if (update_counter % replay_period == 0) & (update_counter >= memory_size):
            
            
            
            sampled_idx, sampled_transitions, ISWeights = memory_buffer.sample(mini_batch_size)
            
            #loss = compute_loss(states, actions, rewards, next_states, done_vals, gamma, q_network, target_q_network)
            
            #取回weight要update tree
            abs_diff = agent_learn(sampled_transitions, ISWeights, gamma)
            
            memory_buffer.batch_update(sampled_idx, abs_diff)
            
            
            if (update_counter % 1000 == 0) & (update_counter >= 100):
                print(gameEnvir.history_score[-100:].mean(), f"; epsilon: {current_epsilon}")
    
        #print(current_epsilon)
    # If the machine successfully meet the standard of the current game play, it's allowed to level up
    
    if (len(gameEnvir.history_score)) >= 201:
        if gameEnvir.history_score[-300:].mean() >= 15.0:
            break

    
    
#%% Here I should implement Pygame code to visualize my result
result = gameEnvir.history_score
w = pd.Series(result).rolling(100).mean().values[49:]
sns.lineplot(x = range(len(w)), y = w )

#%% Saving the model for reloading in the futlrue

q_network.save('saved_model/curriculum_play')