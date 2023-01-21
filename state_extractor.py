#######################################################################
#
# Extract all the valid x and o states with 
# corresponding initial values of each state
#
# Value initialization per state:
#   1. Player wins:             value = 1
#   2. Player loses or tie:     value = 0
#   3. Game continues:          value = 0.5
#
#######################################################################


import copy
import numpy as np
import pickle
from utils import *

# Initialize the memories
state = [0,0,0, 0,0,0, 0,0,0]
x_value = 0.5
o_value = 0.5

x_memory = [[state], [x_value]]
o_memory = [[], []]


def states(state):
    global x_memory
    global o_memory
    
    if sum(state) == 0:
        new_label = 1
    else:
        new_label = -1
    
    legal_moves = [i for i, value in enumerate(state) if value == 0]
    
    for move in legal_moves:
        new_state = copy.copy(state)
        
        new_state[move] = new_label
        
        done = check_done(new_state)
        
        state_is_new = (new_state not in x_memory[0]) and (new_state not in o_memory[0])
        
        if done == 0:
            o_value = 0.5
            x_value = 0.5
            states(new_state)
        elif done == 1:
            o_value = 0
            x_value = 1
        elif done == 2:
            o_value = 0
            x_value = 0
        elif done == -1:
            o_value = 1
            x_value = 0      
        
        if done == 0 and state_is_new:
            if new_label == 1:
                o_memory[0].append(new_state)
                o_memory[1].append(o_value) 
            elif new_label == -1:
                x_memory[0].append(new_state)
                x_memory[1].append(x_value)
                
        elif done != 0 and state_is_new:
            x_memory[0].append(new_state)
            x_memory[1].append(x_value) 
            o_memory[0].append(new_state)
            o_memory[1].append(o_value) 
            
        
       
            
# test_state = [1, -1, 1, -1, 1, -1, 0, 1, 0]

# states(test_state)
# print(x_memory)
# print(o_memory)

states(state)
print(len(x_memory[0]))
print(len(o_memory[0]))
# for i in x_memory[0]:
#     print(i)
for i in range(len(x_memory[0])):
    if x_memory[0][i] == [1, -1, 0, 0, 0, 0, 0, 0, 0]:
        print("AHAHHAA")
        print(i)
    # if 0 not in i:
    #     # print(i)
    #     print('yeey')

# Export the memories as list
with open('./memory/x_state_value_init.txt', 'wb') as file:
    pickle.dump(x_memory, file)
with open('./memory/o_state_value_init.txt', 'wb') as file:
    pickle.dump(o_memory, file)

    
