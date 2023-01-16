import tictactoe
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import copy

env = tictactoe.Game(debug=False)

memory = [[],[],[]]             #[[state], [q values], [n values]]

max_episodes = 500000
eval_rate = 2000

epsilon = 0.2

gamma_memory = []
win_memory = []
memsize_memory = []

win_rate = []
loss_rate = []
draw_rate = []
illegal_rate = []
positive_rate = []

random_methods = ['RANDOM', 'LEAST_CHOSEN', 'UPPER_CONFIDENCE_BOUND']
random_method = random_methods[2]

for episode in range(max_episodes):
    memory_size = len(memory[0])
    print('EPISODE: {}     memory size: {}'.format(episode+1, memory_size))
    memsize_memory.append(memory_size)

    # Initialize the game
    done = False
    state = env.reset()
    
    index_memory = []
    action_memory = []
    reward_memory = []
        
    
    # Start playing
    while not done:        
        deep_state = copy.copy(state)

        try:
            index = memory[0].index(deep_state)
        except Exception as e:
            memory[0].append(deep_state)
            memory[1].append([0, 0, 0, 0, 0, 0, 0, 0, 0])
            memory[2].append([0, 0, 0, 0, 0, 0, 0, 0, 0])
            index = memory_size   
    
        # Get the action based on e-greedy algorithm
        if random.random() > 0.1:
            player = 'ai'
            action = memory[1][index].index(max(memory[1][index]))
        
        else:
            player = 'random'
            if random_method == 'RANDOM':
                action = random.randint(0,8)
            
            elif random_method == 'LEAST_CHOSEN':
                action = memory[2][index].index(min(memory[1][index]))
            
            elif random_method == 'UPPER_CONFIDENCE_BOUND':
                value = 0
                action = 0
                c = 1

                for count, n in enumerate(memory[2][index]):
                    if n == 0:
                        action = count
                        break

                    new_value = memory[1][index][count] + c * np.sqrt(np.log(episode) / n)
                    if new_value >= value:
                        value = new_value
                        action = count
        
        memory[2][index][action] += 1
        
        # Take the game step
        state, reward, done = env.step(action, label = 1)

        # Take a random opponent move
        if done == 0:    
            legal_moves = [i for i, value in enumerate(state) if value == 0]
            random_move = random.choice(legal_moves)
            state, reward, done = env.step(random_move, label = -1)
        
        # Log the memories
        index_memory.append(index)
        action_memory.append(action)
        reward_memory.append(reward)
        
        
        
        
    if episode % eval_rate == 0:
        max_eval_episodes = 200
        winner = []
        for eval_episode in range(max_eval_episodes):

            # Initialize the game
            done = False
            state = env.reset()
            
            # Start playing
            while not done:
                deep_state = copy.copy(state)

                try:
                    index = memory[0].index(deep_state)
                except Exception as e:
                    done = -1
                    break
                
                action = memory[1][index].index(max(memory[1][index]))

                # Take the game step
                state, reward, done = env.step(action, label = 1)

                # Take a random opponent move
                if done == 0:    
                    legal_moves = [i for i, value in enumerate(state) if value == 0]
                    random_move = random.choice(legal_moves)
                    state, reward, done = env.step(random_move, label = -1)
                    
            winner.append(done)
            
        win = 0
        loss = 0
        draw = 0
        illegal = 0
            
        for i in winner:
            if i == -1:
                loss += 1
            elif i == 2:
                draw += 1
            elif i == 1:
                win += 1
            elif i == -2:
                illegal += 1
                
        win_rate.append(win / max_eval_episodes)
        loss_rate.append(loss / max_eval_episodes)
        draw_rate.append(draw / max_eval_episodes)
        illegal_rate.append(illegal / max_eval_episodes)
        positive_rate.append((win+draw)/max_eval_episodes)
        
        
    # Implement Q
    
    for i in range(len(index_memory)):
        index = index_memory[i]
        action = action_memory[i]
        gamma_sum = sum(reward_memory[i:])

        q_prev = memory[1][index][action]
        n = memory[2][index][action]

        # Update the Q value
        memory[1][index][action] = round(q_prev + (gamma_sum-q_prev) / n, 3)

        if i == 0:
            gamma_memory.append(gamma_sum)
            


with open('memory.txt', 'wb') as file:
    pickle.dump(memory, file)



# print('\n', max_episodes, 'games played:')
# print('win:     ,  %{}'.format(win_rate[-1]))
# print('win+draw:     ,  %{}'.format(win_rate[-1]+draw_rate[-1]))
# print('loss:    ,  %{}'.format(loss_rate[-1]))
# print('draw:    {},  %{}'.format(draw, round(100 * draw / max_episodes, 2)))
# print('illegal: {},  %{}'.format(illegal, round(100 * illegal / max_episodes, 2)))

plt.subplot(1,4, 1)
plt.grid()
plt.xlabel('episodes')
plt.ylabel('Memory size')
plt.plot(memsize_memory)

plt.subplot(1,4, 2)
plt.grid()
plt.xlabel('episodes')
plt.ylabel('Win rate')
plt.plot(win_rate[:])

plt.subplot(1,4, 3)
plt.grid()
plt.xlabel('episodes')
plt.ylabel('Win rate')
plt.plot(positive_rate[:])

plt.subplot(1,4, 4)
plt.grid()
plt.xlabel('episodes')
plt.ylabel('Win rate')
plt.plot(illegal_rate[:])
plt.show()
