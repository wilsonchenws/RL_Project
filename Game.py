# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:16:09 2023

@author: wilsonchenws
"""
from Envir import envir, move
import tensorflow as tf
import numpy as np
import pygame
from pygame.locals import QUIT
import sys

#%% Initiate Constants
SIZE_OF_BOARD = 10
WINDOW_LEN = 400
PANNEL_HEIGHT = 100
EPSILON = 0.
#WINDOW_HEIGHT = 800
SQUARE_LENGTH = WINDOW_LEN / SIZE_OF_BOARD
GAMMA = 0.9

FPS = 60
#%% Load AI 
DQN_model = tf.keras.models.load_model('saved_model/curriculum_play')

#%% Main Gain Fuction
def main():
    
    
    #controller.show()


    
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((WINDOW_LEN, WINDOW_LEN + PANNEL_HEIGHT))
    
    #Get player initial position
    y_coord_player = controller.pos_player[0] 
    x_coord_player = controller.pos_player[1]
    
    # Set up the player
    player = pygame.Rect(x_coord_player * SQUARE_LENGTH,
                         y_coord_player * SQUARE_LENGTH + PANNEL_HEIGHT,
                          SQUARE_LENGTH, SQUARE_LENGTH)

    
    # Set up the grid
    grid = []
    for i in range(SIZE_OF_BOARD):
        row = []
        for j in range(SIZE_OF_BOARD):
            rect = pygame.Rect(j * SQUARE_LENGTH, i * SQUARE_LENGTH + PANNEL_HEIGHT, SQUARE_LENGTH, SQUARE_LENGTH)
            row.append(rect)
        grid.append(row)
    
    # Set up the clock
    clock = pygame.time.Clock()
    
    # Set up text for displaying prize
    font = pygame.font.SysFont('Arial', 25)
    font_score = pygame.font.SysFont('Arial', 40)
    
    AI_mode = True
    done = False
    if AI_mode:
        FPS = 3
    else:
        FPS = 60
    
    
    # Game loop
    while not done:
        # Handle events
        clock.tick(FPS)
        
        if AI_mode:
            
            current_state = controller.returnCurrentState()
            action = DQN_model(current_state)
            transition = controller.step(action, EPSILON)         
            done = transition[0, -1]
        for event in pygame.event.get():
            

            
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if AI_mode != True:
                    #current_state, actualAction, reward, next_state, done = controller.step(action, EPSILON)
                    if event.key == pygame.K_UP:
                        transition =  controller.step(move('u'), EPSILON)
                        
                       #print(current_state, actualAction, reward, next_state, done)
                    elif event.key == pygame.K_DOWN:
                        transition  = controller.step(move('d'), EPSILON)
                        #print(current_state, actualAction, reward, next_state, done)
                    elif event.key == pygame.K_LEFT:
                        transition  = controller.step(move('l'), EPSILON)
                        #print(current_state, actualAction, reward, next_state, done)
                    elif event.key == pygame.K_RIGHT:
                        transition  = controller.step(move('r'), EPSILON)
                        #print(current_state, actualAction, reward, next_state, done)
                        
                    done = transition[0,-1]
        
        
        # Draw the grid
        for i in range(SIZE_OF_BOARD):
            for j in range(SIZE_OF_BOARD):
                rect = grid[i][j]
                color = (128, 128, 128) if (i + j) % 2 == 0 else (255, 255, 255)
                pygame.draw.rect(screen, color, rect)
        
        for prize_loc, prize_value in zip(controller.loc_prize, controller.value_of_prize):
            
            y_coord = controller.loc_to_pos(np.array([prize_loc]))[0]
            x_coord = controller.loc_to_pos(np.array([prize_loc]))[1]
            font_size = font.size(str(prize_value))
            screen.blit(font.render(str(prize_value), True, (0,0,0)),
                        (x_coord * SQUARE_LENGTH + SQUARE_LENGTH/2 - font_size[0]/2,
                         y_coord * SQUARE_LENGTH + SQUARE_LENGTH/2- font_size[1]/2 + PANNEL_HEIGHT)
                        )
        
        player.y = controller.pos_player[0] * SQUARE_LENGTH + PANNEL_HEIGHT
        player.x = controller.pos_player[1] * SQUARE_LENGTH

        
        # Draw the player
        pygame.draw.rect(screen, (115, 147, 179), player)
        
        # Draw Total Score
        font_size = font_score.size(str(controller.total_return))
        screen.blit(font_score.render(str(controller.total_return), True, (0,0,0),
                                      (252,252,252)),(0,0))
        
        # Update the screen
        pygame.display.update()
    
        # Tick the clock
        
    controller.reset(SIZE_OF_BOARD ,np.array([SIZE_OF_BOARD * 2]))
    print(controller.history_score)
    main()
    
if __name__ == '__main__':
    
    controller = envir(SIZE_OF_BOARD, np.array([SIZE_OF_BOARD*2]), gamma = GAMMA)
    main()