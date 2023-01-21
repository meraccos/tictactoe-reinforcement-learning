import numpy as np

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

[1, -1, 1, -1, 1, -1, -1, 1, 0]