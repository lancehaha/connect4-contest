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

class AIAgent(object):
    def __init__(self, player_id=1, depth_limit=10):
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
            return 10000, None  # Assign a high positive score for winning
        if is_end(state) or depth == 0:
            return self.evaluate_state(state, self.player_id), None

        max_value = float('-inf')
        best_action = None
        for action in get_valid_col_id(state):
            next_state = step(state.copy(), action, self.player_id, True)
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
            return -10000, None  # Assign a high negative score for losing
        if is_end(state) or depth == 0:
            return self.evaluate_state(state, self.player_id), None

        min_value = float('inf')
        best_action = None
        for action in get_valid_col_id(state):
            next_state = step(state.copy(), action, self.opponent_id, True)
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
                            total = 1 + self.count_consecutive(row + delta_row, col + delta_col, delta_row, delta_col, state, player_id)
                            if total >= 3:  # Priority to block opponent's three in a row
                                score -= 2 * weights.get(total, 0)

        return score

    def count_consecutive(self, row, col, delta_row, delta_col, state, player_id):
        consecutive = 0
        while 0 <= row < 6 and 0 <= col < 7:
            if state[row][col] == player_id:
                consecutive += 1
            else:
                break
            row += delta_row
            col += delta_col
        return consecutive


if __name__ == "__main__":
    board = ConnectFour()
    game = GameControllerPygame(board=board, agents=[HumanPygameAgent(1), AIAgent(2)])
    game.run()