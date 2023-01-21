#######################################################################
#
# A simple human-AI game interface (needs to be improved :))
#
#######################################################################

import tictactoe
import random
import pickle
from utils import *
import time


print('\n\nWelcome to the game!')
print('Good luck!')
time.sleep(1)
print('You will need it :)\n')
time.sleep(1)


env = tictactoe.Game(debug=True)


with open('./memory/state_action.txt', 'rb') as file:
    state_action = pickle.load(file)
    
states = state_action[0]
x_values = state_action[1]
o_values = state_action[2]

score_table = [0, 0]    # human, AI

while True:

    # Initialize the game
    human_player = int(input('\nPlease select the game mode:\n\t -1. Play as O \n\t 1. Play as X\n'))
    
    if human_player == -1:
        ai_label = 1
        human_label = -1
    else:
        ai_label = -1
        human_label = 1
    
    init = True
    
    done = False
    state = env.reset()
    
    # Start playing
    while not done:   
        if (init and human_player == 1):     
            init = False
            random_move = int(input('Please make a move: ')) - 1
            state, reward, done = env.step(random_move, label = human_label)
            continue

        legal_moves = [i for i, value in enumerate(state) if value == 0]
        legal_values = []
        
        for action in legal_moves:
            outcome_indices = possible_outcome_indices(states, state, action)
            
            p = 1 / len(outcome_indices)
            action_value = 0
            
            for i in outcome_indices:
                if human_player == -1:
                    value = x_values[i]
                else:
                    value = o_values[i]
                action_value += p * value          
            
            legal_values.append(round(action_value, 3)) 
        print('legal moves', legal_moves)
        print('legal values', legal_values)

        action = legal_moves[legal_values.index(max(legal_values))]
        print('actiooon', action)
        

        
        ######## Take the game step based on picked action #######
        print('\nSmartPuter is Thinking....')
        time.sleep(1)
        if init:
            action = random.choice([0,2,4,6,8])
            init = False
        state, reward, done = env.step(action, ai_label)

        # Human moves
        if done == 0:    
            init = False
                  
            random_move = int(input('Please make a move: ')) - 1
            state, reward, done = env.step(random_move, label = human_label)
                    
            
    if done == human_player :
        score_table[0] += 1
    elif done in [-1, 1]:
        score_table[1] += 1

    print("\n\nGame Over!")
    time.sleep(1)
    print('AI: ', score_table[1])
    time.sleep(1)
    print('Human: ', score_table[0], '\n\n')
    time.sleep(2)
    

