import numpy as np
import pickle
import copy
from utils import *

with open('./memory/x_state_value_init.txt', 'rb') as file:
    memory = pickle.load(file)
    
states = memory[0]
values = memory[1]



def possible_outcome_indices(state, action):
    next_state = copy.copy(state)
    next_state[action] = 1
    
    indices = []
    
    done = check_done(next_state)

    if done == 0:
        legal_opponent_moves = [i for i, value in enumerate(next_state) if value == 0]
        
        
        for move in legal_opponent_moves:
            possible_state = copy.copy(next_state)
            possible_state[move] = -1
            indices.append(states.index(possible_state))
        
    else:
        indices.append(states.index(next_state))
    
    return indices
    


for i in range(3):
    delta = 0
    for state in states:
        done = check_done(state)

        index = states.index(state)
        
        if done != 0:
            continue
        
        legal_moves = [i for i, value in enumerate(state) if value == 0]
        legal_values = []
        
        for action in legal_moves:
            outcome_indices = possible_outcome_indices(state, action)
            
            p = 1 / len(outcome_indices)
            action_value = 0
            
            for i in outcome_indices:
                value = values[i]
                action_value += p * value          
            
            legal_values.append(round(action_value, 3))  
        
        updated_value = max(legal_values)
        
        delta = max(delta, abs(values[index]-updated_value))

        values[index] = updated_value
    print(delta)
        
        
        
for i in range(len(values)):
    print(states[i], values[i])
        
    