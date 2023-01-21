import tictactoe
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import copy

env = tictactoe.Game(debug=False)

memory = [[],[]]             #[[state], [state value]]
memory_size_log = []
max_episodes = 5000000

epsilon = 0.1

random_methods = ['RANDOM', 'LEAST_CHOSEN', 'UPPER_CONFIDENCE_BOUND']
random_method = random_methods[2]

win = 0
loss = 0
draw = 0
illegal = 0 
nums = 1

def memory_handler(self, state):
    try:
        self.index = self.memory[0].index(state)
    except:
        self.memory[0].append(state)
        self.memory[1].append([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.memory[2].append([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.index = self.memory_size - 1
        
    return index         
    

for episode in range(max_episodes):
    memory_size = len(memory[0])
    memory_size_log.append(memory_size)

    # Initialize the game
    done = False
    state = env.reset()
    
    index_memory = []
    action_memory = []
    reward_memory = []


    if episode < 100000:
        forced_exploration = True
        epsilon = 0.9
    else:
        forced_exploration = False
        epsilon = 0.2

    
    # Start playing
    while not done:        
        deep_state = copy.copy(state)

        # try:
        #     index = memory[0].index(deep_state)
        # except Exception as e:
        #     memory[0].append(deep_state)
        #     memory[1].append([0, 0, 0, 0, 0, 0, 0, 0, 0])
        #     memory[2].append([0, 0, 0, 0, 0, 0, 0, 0, 0])
        #     index = memory_size - 1 
    
        # Get the action based on e-greedy algorithm
        if random.random() > epsilon:
            player = 'ai'
            action = memory[1][index].index(max(memory[1][index]))
        
        else:
            player = 'random'
            if forced_exploration:
                action = memory[2][index].index(min(memory[2][index]))

            elif random_method == 'RANDOM':
                action = random.randint(0,8)
            
            elif random_method == 'LEAST_CHOSEN':
                action = memory[2][index].index(min(memory[1][index]))
            
            elif random_method == 'UPPER_CONFIDENCE_BOUND':
                value = 0
                action = 0
                c = 4

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
    
        
    # Value Iteration
    
    for i in range(len(index_memory)):
        index = index_memory[i]
        action = action_memory[i]
        gamma_sum = sum(reward_memory[i:])

        q_prev = memory[1][index][action]
        n = memory[2][index][action]

        # Update the Q value
        memory[1][index][action] = round(q_prev + (gamma_sum-q_prev) / n, 3)
    
    if player == 'ai':
        nums += 1
        if done == 2:
            draw += 1
        elif done == 1:
            win += 1
        elif done == -1:
            loss += 1
        elif done == -2:
            illegal += 1

    if episode % (max_episodes / 100) == 0:

        print('{} / {} [{}{}]      memory: {} (%{}) win: {}    loss: {}   draw: {}    illegal: {}'.format(str(episode).rjust(len(str(max_episodes)) + 1), 
                                    max_episodes, 
                                    '#'*int(20 * episode / max_episodes), 
                                    '-'*int(20 * (1-(episode+1) / max_episodes)),
                                    str(memory_size).ljust(4),
                                    round(100 * memory_size / 2423, 2),    # max: 2423
                                    round(win / nums, 3)), round(loss / nums, 3), round(draw / nums, 3), round(illegal / nums, 3))
        nums = 1
        wins = 0
            



with open('memory/memory.txt', 'wb') as file:
    pickle.dump(memory, file)
