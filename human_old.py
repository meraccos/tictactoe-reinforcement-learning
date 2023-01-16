import tictactoe_old
import numpy as np
import pickle

env = tictactoe_old.game(debug=True, opponent = 'human')

state = env.reset()
state = np.ndarray.tolist(np.reshape(state, 9))
for i in range(len(state)):
    state[i] = int(state[i])
    
with open('good_memory.txt', 'rb') as file:
    memory = pickle.load(file)
    
done = False

while not done:
    index = -1
    
    # Get the index of the state in the memory
    for count, value in enumerate(memory):
        if value[0] == state:
            index = count
            break
            
    if index == -1:
        print('state does not exist in the memory')

    max_q = 0
    for count, value in enumerate(memory[index][1]):
        if value[0] >= max_q:
            max_q = value[0]
            action = count


    state, reward, done, winner = env.step(action)
    state = np.ndarray.tolist(np.reshape(state, 9))
    state = [int(i) for i in state]

    

