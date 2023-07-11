This is my first project on reinforcement learning.
# Project Purpose
My idea is to train a model that can play a easy game. The game has following rules:

1. The board has 10 x 10 cells.
2. There are 10 prizes on the board with their own value ranging from 1 to 10.
3. Agent can move up, down, right, left at each step.
4. If the agent move to a cell with a prize, it's rewarded with the value of the prize.
5. The agent can move at most 20 steps in each game(episode).

# Model involved
During the development of the project, I've tested out the performances to different famous reinforcement learning models, such as Deep Q-Learning network, Double dueling Q-Learning network, and agent-critic network.
So far, there're no huge improvement by changing different model, and I think the reasons is because of sparse rewards. Therefore the next approaches I might be taking to solve spare reward problem is by:

- Reward Shaping 
- Hieracy Learning
- Curriculum Learning


testing some git control property.