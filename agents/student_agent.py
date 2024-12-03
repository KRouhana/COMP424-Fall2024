# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()

    # Agent Code here:
    best_move = None
    max_depth = 1  # Start with a shallow depth for iterative deepening

    # Iterative deepening search
    while True:
        try:
            # Attempt to find the best move for the current depth
            move, score = self.iterative_deepening(chess_board, player, opponent, max_depth, start_time, 1.98)
            if move is not None:
                best_move = move
        except TimeoutError:
            # Break out of the loop if the time limit is reached
            break

        max_depth += 1  # Increment depth for the next iteration

    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")

    return best_move
  


  def iterative_deepening(self, board, player, opponent, depth, start_time, time_limit):
      """
      Perform minimax search with alpha-beta pruning up to a given depth.
      """
      best_move = None
      best_score = float('-inf')  # Start with the lowest possible score

      valid_moves = get_valid_moves(board, player)
      if not valid_moves:
          return None, 0

      for move in valid_moves:
          # Ensure we don't exceed the time limit
          if time.time() - start_time > time_limit:
              raise TimeoutError("Time limit reached")

          # Simulate the move
          new_board = deepcopy(board)
          execute_move(new_board, move, player)

          # Evaluate the move using the minimax algorithm
          score = self.minimax(new_board, depth - 1, False, player, opponent, float('-inf'), float('inf'), start_time, time_limit)

          # Update the best move if a better score is found
          if score > best_score:
              best_score = score
              best_move = move

      return best_move, best_score

  def minimax(self, board, depth, is_maximizing, player, opponent, alpha, beta, start_time, time_limit):
      """
      Minimax algorithm with alpha-beta pruning.
      """
      # Check time limit
      if time.time() - start_time > time_limit:
          raise TimeoutError("Time limit reached")

      # Check if the game is over or depth is zero
      if depth == 0 or check_endgame(board, player, opponent)[0]:
          return self.evaluate_board(board, player, opponent)

      valid_moves = get_valid_moves(board, player if is_maximizing else opponent)
      if not valid_moves:
          # If no moves are available, evaluate the board
          return self.evaluate_board(board, player, opponent)

      if is_maximizing:
          max_eval = float('-inf')
          for move in valid_moves:
              new_board = deepcopy(board)
              execute_move(new_board, move, player)
              eval = self.minimax(new_board, depth - 1, False, player, opponent, alpha, beta, start_time, time_limit)
              max_eval = max(max_eval, eval)
              alpha = max(alpha, eval)
              if beta <= alpha:
                  break
          return max_eval
      else:
          min_eval = float('inf')
          for move in valid_moves:
              new_board = deepcopy(board)
              execute_move(new_board, move, opponent)
              eval = self.minimax(new_board, depth - 1, True, player, opponent, alpha, beta, start_time, time_limit)
              min_eval = min(min_eval, eval)
              beta = min(beta, eval)
              if beta <= alpha:
                  break
          return min_eval

  def evaluate_board(self, board, player, opponent):
      """
      Heuristic evaluation function for the board.
      """
      # Count pieces and prioritize corners
      score = np.sum(board == player) - np.sum(board == opponent)

      # Add weight for corner stability
      corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
      for corner in corners:
          if board[corner] == player:
              score += 10
          elif board[corner] == opponent:
              score -= 10

      return score

