#######################################################################
#
# Extract all the valid x and o states with 
# corresponding initial values of each state
#
# Value initialization per state:
#   1. Player wins:             value = 1
#   2. Player loses:            value = -1
#   3. Tie:                     value = 0
#   4. Game continues:          value = 0.5
#
#######################################################################

import copy
import pickle
from utils import *

# Initialize the memories
state = [0,0,0, 0,0,0, 0,0,0]
x_value = 0.5
o_value = 0.5

memory = [[state], [x_value], [o_value]]

def states(state):
    global memory
    
    if sum(state) == 0:
        new_label = 1
    else:
        new_label = -1
    
    legal_moves = [i for i, value in enumerate(state) if value == 0]
    
    for move in legal_moves:
        new_state = copy.copy(state)
        
        new_state[move] = new_label
        
        if new_state not in memory[0]:
            
            done = check_done(new_state)
            nonzero = len([i for i, value in enumerate(new_state) if value != 0])
        
            if done == 0:
                o_value = 0.5
                x_value = 0.5
                states(new_state)
            elif done == 1:
                o_value = -1
                x_value = 1
            elif done == 2:
                o_value = 0
                x_value = 0
            elif done == -1:
                o_value = 1
                x_value = -1     
            
            memory[0].append(new_state)
            memory[1].append(x_value)
            memory[2].append(o_value)
            
        
states(state)

# Export the memories as list
with open('./memory/state_value_init.txt', 'wb') as file:
    pickle.dump(memory, file)

print('States successfully exported with initial values!')
print('Size of memory: ', len(memory[0]))
