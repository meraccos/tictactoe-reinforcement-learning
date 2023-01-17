import numpy as np


class Game():
    def __init__(self, debug=False):  
        self.done = 0                   # 0 for not done, 1 for X wins, -1 for O wins, -2 for illegal move, 2 for draw
        self.debug = debug
        self.reset()
        if self.debug:
            self._draw()

    def _draw(self):
        sign_board = list(' ' * (self.board[i] == 0) +
                          'X' * (self.board[i] == 1) + 
                          'O' * (self.board[i] == -1) for i in range(9))
            
        print('-------------')
        print('| {} | {} | {} |'.format(*sign_board[0:3]))
        print('| {} | {} | {} |'.format(*sign_board[3:6]))
        print('| {} | {} | {} |'.format(*sign_board[6:9]))
        print('-------------')

    def _computeDone(self):
        board_matrix = np.array(self.board).reshape((3, 3))
        sum_v = np.sum(board_matrix, axis=0)
        sum_h = np.sum(board_matrix, axis=1)
        checker = np.concatenate((sum_v, sum_h, 
                                  [np.trace(board_matrix), 
                                   np.trace(np.fliplr(board_matrix))]))

        x_wins = 3 in checker
        o_wins = -3 in checker
        draw = 0 not in self.board

        if x_wins:
            self.done = 1      
        elif o_wins:
            self.done = -1
        elif draw:
            self.done = 2
        else:
            self.done = 0

    def _computeReward(self):
        if self.done == 0:            # game continues
            # self.reward = 0.1
            self.reward = -0.1
            
        elif self.done == 1:          # X wins
            self.reward = 1
            if self.debug:
                print('\n\n\n~~~~~ X wins! ~~~~~\n\n')
                
        elif self.done == 2:          # draw
            self.reward = 0.5
            if self.debug:
                print('\n\n\n~~~~~ draw! ~~~~~\n\n')
                
        elif self.done == -1:         # O wins
            self.reward = -0.5
            if self.debug:
                print('\n\n\n~~~~~ O wins! ~~~~~\n\n')
                
        elif self.done == -2:         # illegal move
            self.reward = -1
            if self.debug:
                print('\n\n\n~~~~~ illegal! ~~~~~\n\n')

    def step(self, action, label):
        # Check for the illegal move
        if self.board[action] != 0:
            self.done = -2
        else:
            self.board[action] = label
            self._computeDone()

        self._computeReward()
        
        if self.debug:
            self._draw()

        return self.board, self.reward, self.done


    def reset(self):
        self.board = [0,0,0,0,0,0,0,0,0]
        return self.board
