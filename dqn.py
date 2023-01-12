# -*- coding: utf-8 -*-
import time
import datetime
import random
import tictactoe
import tensorflow as tf
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout
# from keras.backend import reshape
# from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

EPISODES = 100000

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        # numCells = 9
        # outcomes = 9
        # model = Sequential()
        # model.add(Dense(200, activation='relu', input_shape=(9, )))
        # model.add(Dropout(0.2))
        # model.add(Dense(125, activation='relu'))
        # model.add(Dense(75, activation='relu'))
        # model.add(Dropout(0.1))
        # model.add(Dense(25, activation='relu'))
        # model.add(Dense(outcomes, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[tensorboard_callback])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = tictactoe.game()

    state_size = 9
    action_size = 9
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    start_time = time.time()
    end_time = time.time()

    for e in range(EPISODES):
        reward_sum = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for times in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            reward_sum += reward
            if done:
                print("episode: {}/{},   length: {},   e: {:.2},  r: {},   duration: {},  time: {}s"
                      .format(e, EPISODES, times, agent.epsilon, round(reward_sum, 2), round(time.time()-end_time, 2), round(time.time()-start_time, 2)))
                end_time = time.time()
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 15 == 0:
            agent.save("./save/tictactoe-dqn.h5")
        
        # Logging
        
        with open('log_reward.txt', 'a') as file:
            file.write(str(reward_sum))
            file.write('\n')
        with open('log_length.txt', 'a') as file:
            file.write(str(times))
            file.write('\n')
        with open('log_winner.txt', 'a') as file:
            file.write(str(env.winner))
            file.write('\n')
        