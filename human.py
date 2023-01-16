import tictactoe
import numpy as np
import pickle
import random

env = tictactoe.Game(debug=True)
state = env.reset()
done = False
    
with open('best_memory.txt', 'rb') as file:
    memory = pickle.load(file)

# print('debug: ', memory[0][0], memory[1][0], memory[2][0])

first_move = True

while not done:
    index = -1
    
    # Get the index of the state in the memory
    for count, value in enumerate(memory[0]):
        if value == state:
            index = count
            break
            
    if index == -1:
        print('state does not exist in the memory')

    if first_move:
        good_qs = []
        for count, value in enumerate(memory[1][index]):
            if value >= 0.3:
                good_qs.append(count)
        
        action = random.choice(good_qs)
        first_move = False
    
    else:
        max_q = 0
        for count, value in enumerate(memory[1][index]):
            if value >= max_q:
                max_q = value
                action = count


    state, reward, done = env.step(action, label = 1)

    if done == 0:
        done = -2
        while done == -2:
            action = int(input('Please make the move: (O)\t'))
            state, reward, done = env.step(action-1, label = -1)
    else:
        print('\n\n')
    
    
    
    
    
    # def _opponentMove(self):
    #     legal_moves = np.where(self.board == 0)[0]
    #     if self.opponent == 'random':
    #         action = random.choice(legal_moves)
        
    #     elif self.opponent == 'human':
    #         action = -1
    #         while action not in legal_moves:
    #             action = int(input('Please make the move: (O)\t')) - 1 
        
    #     return action

