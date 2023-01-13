import tictactoe
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

env = tictactoe.game(debug=False, opponent = 'random')

# memory = [[[0] * 9, [1] * 9],[[2] * 9, [3] * 9]]     # [0]: state, [1]: Q values associated to 9 actions
memory = []
max_episodes = 2000000

epsilon = 0.2
gamma = 0.99

gamma_memory = []
win_memory = []
mem_memory = []

for episode in range(max_episodes):
    print('EPISODE: {}     memory size: {}'.format(episode+1, len(memory)), end = '\t')
    # Initialize the game
    done = False
    state = env.reset()
    state = np.ndarray.tolist(np.reshape(state, 9))
    for i in range(len(state)):
        state[i] = int(state[i])
    
    index_memory = []
    action_memory = []
    reward_memory = []

    mem_memory.append(len(memory))
    
    # Start playing
    while not done:
        index = -1
        
        # Get the index of the state in the memory
        for i in range(len(memory)):
            if memory[i][0] == state:
                index = i
                
        if index == -1:
            memory.append([state, [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]])
            index = len(memory) - 1
        
        
        # Get the action based on e-greedy algorithm
        if random.random() > 0.1:
            max_q = 0
            action = 0

            # Force to explore unexplored
            for i in range(9):
                if memory[index][1][i][0] == 0:
                    action = i
                    break

            for i in range(9):
                if memory[index][1][i][0] >= max_q:
                    max_q = memory[index][1][i][0]
                    action = i
        else:
            # # Force to pick the one least chosen
            # min_n = 100
            # action = 0

            # for i in range(9):
            #     if memory[index][1][i][1] <= min_n and memory[index][1][i][0] != -1:
            #         min_n = memory[index][1][i][1]
            #         action = i


            action = random.randint(0,8)
        
        memory[index][1][action][1] += 1

        # Take the game step
        state, reward, done, winner = env.step(action)
        state = np.ndarray.tolist(np.reshape(state, 9))
        for i in range(len(state)):
            state[i] = int(state[i])
        
        # Log the memories
        index_memory.append(index)
        action_memory.append(action)
        reward_memory.append(reward)

        if done:
            if reward_memory[-1] == -1:
                print('illegal')
            win_memory.append(winner)

    # Implement Q
    for i in range(len(index_memory)):
        gamma_sum = 0
        for j in range(len(reward_memory[i:])):
            reward = reward_memory[i+j]
            power = j
            gamma_sum += reward * gamma ** power

        q_prev = memory[index_memory[i]][1][action_memory[i]][0]
        n = memory[index_memory[i]][1][action_memory[i]][1]

        memory[index_memory[i]][1][action_memory[i]][0] = round(q_prev + (gamma_sum-q_prev) / n, 3)

        if i == 0:
            gamma_memory.append(gamma_sum)

with open('memory.txt', 'wb') as file:
    pickle.dump(memory, file)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


win = 0
loss = 0
draw = 0
illegal = 0

for i in win_memory:
    if i == -1:
        loss += 1
    elif i == 0:
        draw += 1
    elif i == 1:
        win += 1
    elif i == -2:
        illegal += 1

print('\n', max_episodes, 'games played:')
print('win: {},  %{}'.format(win, round(100 * win / (win + loss + draw + illegal),2)))
print('loss: {},  %{}'.format(loss, round(100 * loss / (win + loss + draw + illegal),2)))
print('draw: {},  %{}'.format(draw, round(100 * draw / (win + loss + draw + illegal),2)))
print('illegal: {},  %{}'.format(illegal, round(100 * illegal / (win + loss + draw + illegal),2)))

plt.subplot(1,3, 1)
plt.plot(smooth(gamma_memory,5))
plt.subplot(1,3, 2)
plt.plot(smooth(win_memory,5))
plt.subplot(1,3, 3)
plt.plot(smooth(mem_memory,5))
plt.show()
