import numpy as np
import random
import math

class game():
    def __init__(self, debug, opponent):
        self.board = np.zeros((3,3))
        self.debug = debug
        self.opponent = opponent    # human or random

    def _draw(self):
        draw_board = ['-'] * 9
        for i in range(9):
            value = self.board[math.floor(i /3)][i % 3]
            draw_board[i] = 'X' * (value == 1) + 'O' * (value == -1) + ' ' * (value == 0)
             
        print('-------------------')
        for i in range(3):
            print('| ', draw_board[3*i], ' | ', draw_board[3*i+1], ' | ',  draw_board[3*i+2], ' |')
        print('-------------------')
    
    def _computerMove(self, action):
        legal_moves = []
        for i in range(9):
            if self.board[math.floor(i /3)][i % 3] == 0:
                legal_moves.append(i)
        
        move = random.choice(legal_moves)
        self.board[math.floor(move /3)][move % 3] = -1
        
        if self.debug:
            print('Agent moves: {}  Computer moves: {}'.format(action+1, move+1))
            

    def _computeDone(self, action):
        # Check if the move is legal
        if (action != -1) and (self.board[math.floor(action /3)][action % 3] != 0):
            if self.debug:
                print('illegal move: {}!\n'.format(action + 1))
            return [1, -2]  # illegal move
        
        # Check if the game is over
        sum_v = np.sum(self.board, axis = 0)
        sum_h = np.sum(self.board, axis = 1)

        x_wins = (3 in sum_v) or (3 in sum_h)
        o_wins = (-3 in sum_v) or (-3 in sum_h)
        
        if (x_wins or o_wins) and self.debug:
            self._draw()
            
        if x_wins:
            print('X WINS!')
            return [1, 1]   # done, 1 wins    
        elif o_wins:
            print('O WINS!')
            return [1, -1]   # done, -1 wins
        elif 0 in self.board:
            return [0, 0]   # not done
        else:
            print('draw')
            return [1, 0]   # done, draw
        
        
    
    def step(self, action):
        # Check if the move is illegal
        
        done, self.winner = self._computeDone(action)
        if not done:
            self.board[math.floor(action /3)][action % 3] = 1
            done, self.winner = self._computeDone(-1)
        
        if self.debug:
            self._draw()
        
        if not done:
            if self.opponent == 'human':
                move = int(input('Please make your move: (O)\t'))-1
                self.board[math.floor(move /3)][move % 3] = -1 
            
            elif self.opponent == 'random':
                self._computerMove(action)
            
            reward = 0.2
            
            done, self.winner = self._computeDone(-1)
            
        if done:
            if self.winner == -2:   
                reward = -1     # illegal
            elif self.winner == 1:
                reward = 2      # X wins
            elif self.winner == -1:
                reward = -0.5   # O wins    
            elif self.winner == 0:
                reward = 0.5    # draw
        return self.board, reward, done, self.winner

    def reset(self):
        self.board = np.zeros((3,3))
        return self.board