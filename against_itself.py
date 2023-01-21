#######################################################################
#
# Retrieve the evaluated state-value pairs and 
# let the trained agent play against itself.
# The moves are greedy, thus the game is deterministic.
#
#######################################################################

import tictactoe
import pickle
from utils import *


#### Load the state-value pairs ######
with open('./memory/state_action.txt', 'rb') as file:
    state_action = pickle.load(file)
    
states = state_action[0]
x_values = state_action[1]
o_values = state_action[2]


#### Initialize the game ######
env = tictactoe.Game(debug=True)
state = env.reset()
done = False


while not done:       
    #### Determine the player ######
    if sum(state) == 0:
        label = 1
        values = x_values 
    else:
        label = -1
        values = o_values 

    #### Evaluate the best action according to MDP ######
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
    

    ######## Take the game step based on the picked action #######
    state, reward, done = env.step(action, label = label)


