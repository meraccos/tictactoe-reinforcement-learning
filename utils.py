import numpy as np
import copy

def check_done(state):
    # Checks if the state is terminal (done) and 
    # returns value based on the condition of termination
    
    board_matrix = np.array(state).reshape((3, 3))
    sum_v = np.sum(board_matrix, axis=0)
    sum_h = np.sum(board_matrix, axis=1)
    checker = np.concatenate((sum_v, sum_h, 
                                [np.trace(board_matrix), 
                                np.trace(np.fliplr(board_matrix))]))

    x_wins = 3 in checker
    o_wins = -3 in checker
    draw = 0 not in state
    
    if x_wins:
        done = 1      
    elif o_wins:
        done = -1
    elif draw:
        done = 2
    else:
        done = 0
    
    return done


def possible_outcome_indices(states, state, action):
    next_state = copy.copy(state)
    player = (sum(state) == 0)
    if player:
        next_state[action] = 1
    else:
        next_state[action] = -1
    
    indices = []
    
    done = check_done(next_state)

    if done == 0:
        legal_opponent_moves = [i for i, value in enumerate(next_state) if value == 0]
        
        
        for move in legal_opponent_moves:
            possible_state = copy.copy(next_state)
            
            if player:
                possible_state[move] = -1
            else:
                possible_state[move] = 1
            indices.append(states.index(possible_state))
        
    else:
        indices.append(states.index(next_state))
    
    return indices