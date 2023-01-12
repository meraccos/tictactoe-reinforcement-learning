import tictactoe
import numpy as np
import pickle

env = tictactoe.game(debug=True, opponent = 'human')

state = env.reset()
state = np.ndarray.tolist(np.reshape(state, 9))
for i in range(len(state)):
    state[i] = int(state[i])

done = False

with open('good_memory.txt', 'rb') as file:
    memory = pickle.load(file)
print(memory)
while not done:

    index = -1
    
    # Get the index of the state in the memory
    for i in range(len(memory)):
        if memory[i][0] == state:
            index = i
            
    if index == -1:
        print('state does not exist in the memory')

    max_q = 0
    action = 0
    for i in range(9):
        # Force to explore unexplored
        if memory[index][1][i][0] == 0:
            action = i
            break

        if memory[index][1][i][0] >= max_q:
            max_q = memory[index][1][i][0]
            action = i


    state, reward, done, winner = env.step(action)
    state = np.ndarray.tolist(np.reshape(state, 9))
    for i in range(len(state)):
        state[i] = int(state[i])
    

