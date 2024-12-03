from game_utils import *
from pygame_simulator import GameControllerPygame, HumanPygameAgent
from connect_four import ConnectFour

c4_board = initialize()
print(c4_board.shape)

print(c4_board)

get_valid_col_id(c4_board)

step(c4_board, col_id=2, player_id=1, in_place=True)

print(c4_board)

step(c4_board, col_id=2, player_id=2, in_place=True)
step(c4_board, col_id=2, player_id=1, in_place=True)
step(c4_board, col_id=2, player_id=2, in_place=True)
step(c4_board, col_id=2, player_id=1, in_place=True)
step(c4_board, col_id=2, player_id=2, in_place=True)
print(c4_board)

print(get_valid_col_id(c4_board))

step(c4_board, col_id=2, player_id=1, in_place=True)

class ZeroAgent(object):
    def __init__(self, player_id=1):
        pass
    def make_move(self, state):
        return 0

# Step 1
agent1 = ZeroAgent(player_id=1) # Yours, Player 1
agent2 = ZeroAgent(player_id=2) # Opponent, Player 2

# Step 2
contest_board = initialize()

# Step 3
p1_board = contest_board.view()
p1_board.flags.writeable = False
move1 = agent1.make_move(p1_board)

# Step 4
step(contest_board, col_id=move1, player_id=1, in_place=True)

from simulator import GameController, HumanAgent
from connect_four import ConnectFour

board = ConnectFour()
game = GameController(board=board, agents=[HumanAgent(1), HumanAgent(2)])
game.run()

## Task 1.1 Make a valid move

import torch
import torch.nn as nn
import numpy as np
from game_utils import *  # Assuming this imports functions like is_win, is_end, step, get_valid_col_id

class Connect4OutcomePredictor(nn.Module):
    def __init__(self):
        super(Connect4OutcomePredictor, self).__init__()
        self.fc1 = nn.Linear(43, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output layer with 3 classes (win, loss, draw)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class AIAgent(object):
    def __init__(self, player_id=2, depth_limit=4):
        self.player_id = player_id
        self.opponent_id = 2 if player_id == 1 else 1
        self.depth_limit = depth_limit

        model = Connect4OutcomePredictor()
        
        # fc1 weights and biases
        fc1_weights = np.array([[-0.05894192,  0.0563272, 0.13090901, ...],  # Fill in the full array
                                [-0.01428001, -0.01777992, 0.0970642, ...],  # This is just an example
                                # Add all rows for fc1 weights
                            ])

        fc1_bias = np.array([0.01, -0.02, 0.03, ..., 0.05])  # Fill in with all the bias values for fc1

        # fc2 weights and biases
        fc2_weights = np.array([[-0.075832, -0.06770291, -0.06400633, ...],  # Fill in all rows
                                # Add all rows for fc2 weights
                            ])

        fc2_bias = np.array([0.01, 0.02, -0.03, ..., 0.05])  # Fill in with all the bias values for fc2

        # fc3 weights and biases
        fc3_weights = np.array([[0.00398379,  0.08683397,  0.04730075, ...],  # Fill in all rows
                                # Add all rows for fc3 weights
                            ])

        fc3_bias = np.array([0.01, -0.01, 0.02])  # Fill in with all the bias values for fc3

        # Assign the weights and biases to the model's layers

        with torch.no_grad():  # Ensures that the gradient computation is disabled while updating weights
            # Convert the numpy arrays to tensors and assign
            model.fc1.weight = nn.Parameter(torch.tensor(fc1_weights, dtype=torch.float32))
            model.fc1.bias = nn.Parameter(torch.tensor(fc1_bias, dtype=torch.float32))

            model.fc2.weight = nn.Parameter(torch.tensor(fc2_weights, dtype=torch.float32))
            model.fc2.bias = nn.Parameter(torch.tensor(fc2_bias, dtype=torch.float32))

            model.fc3.weight = nn.Parameter(torch.tensor(fc3_weights, dtype=torch.float32))
            model.fc3.bias = nn.Parameter(torch.tensor(fc3_bias, dtype=torch.float32))

        # Set model to evaluation mode
        model.eval()
        

    def make_move(self, state):
        if is_end(state):
            return None
        _, action = self.minimax_search(state, self.depth_limit)
        return action

    def minimax_search(self, state, depth):
        value, action = self.max_value(state, depth, float('-inf'), float('inf'))
        return value, action

    def max_value(self, state, depth, alpha, beta):
        if is_win(state):
            return -10000, None  # Assign a high positive score for winning
        if is_end(state) or depth == 0:
            return self.evaluate_state(state, self.player_id), None

        max_value = float('-inf')
        best_action = None
        for action in get_valid_col_id(state):
            next_state = step(state, action, self.player_id, False)
            value, _ = self.min_value(next_state, depth - 1, alpha, beta)
            if value > max_value:
                max_value = value
                best_action = action
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return max_value, best_action

    def min_value(self, state, depth, alpha, beta):
        if is_win(state):
            return 10000, None  # Assign a high negative score for losing
        if is_end(state) or depth == 0:
            return self.evaluate_state(state, self.player_id), None

        min_value = float('inf')
        best_action = None
        for action in get_valid_col_id(state):
            next_state = step(state, action, self.opponent_id, False)
            value, _ = self.max_value(next_state, depth - 1, alpha, beta)
            if value < min_value:
                min_value = value
                best_action = action
            beta = min(beta, value)
            if beta <= alpha:
                break
        return min_value, best_action

    def evaluate_state(self, state, player_id):
        # Directly flatten the board state since it is already in numerical form
        flattened_board = state.flatten().tolist()  # Use NumPy flatten to convert to 1D list

        # Add player ID to the input
        player_id_value = 0 if player_id == 1 else 1
        flattened_board.append(player_id_value)

        # Convert the input to a PyTorch tensor and make a prediction
        x_input = torch.tensor(flattened_board, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Use the trained model to evaluate the current board state
        with torch.no_grad():
            output = self.model(x_input)  # Get raw logits from the model
            probabilities = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
            win_prob, draw_prob, lose_prob = probabilities[0]

        # Calculate a score based on the probabilities
        # Higher weight for winning and avoiding losing
        score = (win_prob * 1000) - (lose_prob * 1000) + (draw_prob * 500)
        
        return score






def test_task_1_1():
    from utils import check_step, actions_to_board
    
    # Test case 1
    res1 = check_step(ConnectFour(), 1, AIAgent)
    assert(res1 == "Pass")
    
    # Test case 2
    res2 = check_step(actions_to_board([0, 0, 0, 0, 0, 0]), 1, AIAgent)
    assert(res2 == "Pass")
    
    # Test case 3
    res2 = check_step(actions_to_board([4, 3, 4, 5, 5, 1, 4, 4, 5, 5]), 1, AIAgent)
    assert(res2 == "Pass")

## Task 2.1: Defeat the Baby Agent
from game_utils import * 
import numpy as np 

class AIAgent(object):
    def __init__(self, player_id=2, depth_limit=4):
        self.player_id = player_id
        self.opponent_id = 2 if player_id == 1 else 1
        self.depth_limit = depth_limit

    def make_move(self, state):
        if is_end(state):
            return None
        _, action = self.minimax_search(state, self.depth_limit)
        return action

    def minimax_search(self, state, depth):
        value, action = self.max_value(state, depth, float('-inf'), float('inf'))
        return value, action

    def max_value(self, state, depth, alpha, beta):
        if is_win(state):
            return -10000, None  # Assign a high positive score for winning
        if is_end(state) or depth == 0:
            return self.evaluate_state(state, self.player_id), None

        max_value = float('-inf')
        best_action = None
        for action in get_valid_col_id(state):
            next_state = step(state, action, self.player_id, False)
            value, _ = self.min_value(next_state, depth - 1, alpha, beta)
            if value > max_value:
                max_value = value
                best_action = action
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return max_value, best_action

    def min_value(self, state, depth, alpha, beta):
        if is_win(state):
            return 10000, None  # Assign a high negative score for losing
        if is_end(state) or depth == 0:
            return self.evaluate_state(state, self.player_id), None

        min_value = float('inf')
        best_action = None
        for action in get_valid_col_id(state):
            next_state = step(state, action, self.opponent_id, False)
            value, _ = self.max_value(next_state, depth - 1, alpha, beta)
            if value < min_value:
                min_value = value
                best_action = action
            beta = min(beta, value)
            if beta <= alpha:
                break
        return min_value, best_action

    def evaluate_state(self, state, player_id):
        opponent_id = 2 if player_id == 1 else 1
        score = 0

        # Parameters for scoring
        weights = {2: 10, 3: 50}  # Adjust weights for two and three in a row
        center_column_index = 3
        center_bonus = 3  # Bonus for center column piece

        # Directions for line checking
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]


        # Check all positions on the board
        for row in range(6):
            for col in range(7):
                if state[row][col] == player_id:
                    # Score center column more highly
                    if col == center_column_index:
                        score += center_bonus

                    # Check each direction for scoring potential
                    for delta_row, delta_col in directions:
                        if row + 3 * delta_row < 6 and col + 3 * delta_col < 7 and row + 3 * delta_row >= 0 and col + 3 * delta_col >= 0:
                            total = 1 + self.count_consecutive(row + delta_row, col + delta_col, delta_row, delta_col, state, player_id)
                            if total >= 2:
                                score += weights.get(total, 0)

                # Consider opponent pieces to block
                if state[row][col] == opponent_id:
                    # Check each direction for blocking opponent's scoring potential
                    for delta_row, delta_col in directions:
                        if row + 3 * delta_row < 6 and col + 3 * delta_col < 7 and row + 3 * delta_row >= 0 and col + 3 * delta_col >= 0:
                            total = 1 - self.count_consecutive(row + delta_row, col + delta_col, delta_row, delta_col, state, opponent_id)
                            if total >= 3:  # Priority to block opponent's three in a row
                                score -= 2 * weights.get(total, 0)

        return score

    def count_consecutive(self, row, col, delta_row, delta_col, state, player_id):
        consecutive = 0
        counter = 0
        while 0 <= row < 6 and 0 <= col < 7 and counter < 4:
            if state[row][col] == player_id:
                consecutive += 1
            elif state[row][col] == 0:
                consecutive += 0
            else:
                break
            row += delta_row
            col += delta_col
            counter += 1
        return consecutive


def test_task_2_1():
    assert(True)
    # Upload your code to Coursemology to test it against our agent.

## Task 2.2: Defeat the Base Agent

from game_utils import * 
import numpy as np 
 
 
class AIAgent(object): 
     
    def init(self, player_id): 
         
        self.max_depth = 4 
        self.player_id = player_id 
        self.opponent_id = 2 if player_id == 1 else 1 
        #self.visited = {} 
 
    def make_move(self, state): 
 
        if is_end(state): 
            return None 
        max_depth = self.max_depth 
        best_value, best_input_col = self.minimax_alpha_beta(state, max_depth) 
 
        return best_input_col 
     
    # only if the four contains 0 and player_id piece 
    def count_four(self, state, piece_id, r, c, r_dir, c_dir): 
         
        ROW, COL = state.shape 
        same_piece = 0 
        for i in range(4): 
            if state[r][c] == piece_id: 
                same_piece += 1 
            elif state[r][c] == 0: 
                same_piece += 0 
            else: 
                same_piece = 0 
                break 
            r += r_dir 
            c += c_dir 
        return same_piece 
         
    # rewards the middle col 
    # count how many player_id piece or opponent_id piece in the "four" 
    # consider 4 directions [horizontal, vertical, top left bottom right, top right bottom left] 
    def evaluate(self, state): 
          
        player_id = self.player_id 
        opponent_id = self.opponent_id 
        ROW, COL = state.shape 
 
        score = 0 
        middle_col = COL // 2 
        directions = [(0,1), (1,0), (1,1), (1,-1)] 
         
        for r in range(ROW): 
            for c in range(COL): 
                # this piece is AI 
                if state[r][c] == player_id: 
 
                    for r_dir, c_dir in directions: 
                        if r + 3*r_dir < ROW and c + 3*c_dir < COL and c + 3*c_dir >= 0: 
                            count = self.count_four(state, player_id, r, c, r_dir, c_dir) 
                            if count == 2: 
                                score += 10 
                            if count == 3: 
                                score += 50 
                                 
                    if c == middle_col: 
                        score += 3 
                             
                # opponent piece 
                elif state[r][c] == opponent_id: 
                     
                    for r_dir, c_dir in directions: 
                        if r + 3*r_dir < ROW and c + 3*c_dir < COL and c + 3*c_dir >= 0: 
                            count = self.count_four(state, opponent_id, r, c, r_dir, c_dir) 
                            if count == 3: # opponent about to win, punish more 
                                score -= 60 
 
        return score 
 
    def is_terminate(self, state): 
        return len(get_valid_col_id(state)) == 0 
 
 
    # minimax with alpha beta pruning & depth limit 
    def minimax_alpha_beta(self, state, max_depth): 
        alpha = -float('inf') 
        beta = float('inf') 
        best_value, best_input_col = self.maximize(state, alpha, beta, max_depth) 
        return (best_value, best_input_col) 
     
    def maximize(self, state, alpha, beta, depth): 
 
        if is_win(state): 
            return (-100000, None)  # -100000 because MIN wins, MAX loses 
         
        if self.is_terminate(state) or depth == 0: 
            result = (self.evaluate(state), None) 
            return result 
 
        best_value = -float('inf') 
        best_input_col = None 
        valid_input_cols = get_valid_col_id(state) 
             
        for input_col in valid_input_cols : 
                 
            next_state = step(state, input_col, self.player_id, False) 
            value, _ = self.minimize(next_state, alpha, beta, depth - 1) 
                   
            if (value > best_value):   # max(v, min()) in lecture
                best_value = value 
                best_input_col = input_col 
 
            #pruning 
            alpha = max(alpha, best_value) 
            if (best_value >= beta) : 
                break 
 
        return (best_value, best_input_col)     
 
    def minimize(self, state, alpha, beta, depth): 
             
        if is_win(state): 
            return (100000, None)  # 100000 because MIN loses, MAX wins 
             
        if self.is_terminate(state) or depth == 0: 
            result = (self.evaluate(state), None) 
            return result 
 
        best_value = float('inf') 
        best_input_col = None 
        valid_input_cols = get_valid_col_id(state) 
             
        for input_col in valid_input_cols : 
                 
            next_state = step(state, input_col, self.opponent_id, False)  
            value, _ = self.maximize(next_state, alpha, beta, depth - 1)                 
                 
            if (value < best_value) : # min(v, max()) in lecture 
                best_value = value 
                best_input_col = input_col 
                
            #pruning 
            beta = min(beta, best_value) 
            if (best_value <= alpha): 
                break                 
             
        return (best_value, best_input_col)

def test_task_2_2():
    assert(True)
    # Upload your code to Coursemology to test it against our agent.


if __name__ == '__main__':
    test_task_1_1()
    test_task_2_1()
    test_task_2_2()