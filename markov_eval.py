#######################################################################
#
# Retrieve the evaluated state-value pairs and 
# play (X) according to MDP agains a random player (O). 
# Evaluate the results.
#
#######################################################################

import tictactoe
import random
import pickle
from utils import *

env = tictactoe.Game(debug=False)

max_episodes = 1000

win = 0
loss = 0
draw = 0
illegal = 0 
nums = 1


with open('./memory/state_action.txt', 'rb') as file:
    state_action = pickle.load(file)
    
states = state_action[0]
values = state_action[1]


for episode in range(max_episodes):

    # Initialize the game
    done = False
    state = env.reset()
    
    # Start playing
    while not done:        

        ####### Find the legal moves and evaluate the best action according to MDP
        legal_moves = [i for i, value in enumerate(state) if value == 0]
        legal_values = []
        
        for action in legal_moves:
            outcome_indices = possible_outcome_indices(states, state, action)
            
            p = 1 / len(outcome_indices)
            action_value = 0
            
            for i in outcome_indices:
                value = values[i]
                action_value += p * value          
            
            legal_values.append(round(action_value, 3)) 
        
        action = legal_moves[legal_values.index(max(legal_values))]

        
        ######## Take the game step based on picked action #######
        state, reward, done = env.step(action, label = 1)

        ######## Take a random opponent move #####################
        if done == 0:    
            legal_moves = [i for i, value in enumerate(state) if value == 0]
            random_move = random.choice(legal_moves)
            state, reward, done = env.step(random_move, label = -1)
    
    
    ##### Log the results ##############
    nums += 1
    if done == 2:
        draw += 1
    elif done == 1:
        win += 1
    elif done == -1:
        loss += 1



print('Episodes played: {} \nwin: {} \nloss: {} \ndraw: {}'.format( 
                        max_episodes, round(win / nums, 3), round(loss / nums, 3), round(draw / nums, 3)))

        



