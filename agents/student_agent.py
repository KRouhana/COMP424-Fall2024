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

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):
    start_time = time.time()
    
    # Get valid moves
    valid_moves = get_valid_moves(chess_board, player)
    if not valid_moves:
        return None
    
    best_move = valid_moves[0] # Default to first valid move
    depth = 1 # Start with depth 1
    
    # Iterative deepening - increase depth until time runs out
    try:
        while time.time() - start_time < 1.99:  # Leave 1ms buffer
            move, score = self.iterative_deepening(chess_board, player, opponent, 
                                                 depth, start_time, 1.99)
            if move is not None:
                best_move = move
            depth += 1
    except TimeoutError:
        pass
        
    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")
    return best_move

  def evaluate_board(self, board, player, opponent):
    """
        Board evaluation using multiple weighted heuristics:
        1. Endgame: Pure piece count difference with high weight
        2. Corner control: Most important strategic positions
        3. Edge stability: Second most important positions
        4. Mobility: Number of available moves
        5. Piece differential: Least important in early/midgame
    """
    # Check for game end - use pure piece count
    if check_endgame(board, player, opponent)[0]:
        player_count = np.sum(board == player)
        opponent_count = np.sum(board == opponent) 
        return 1000 * (player_count - opponent_count)

    score = 0
    size = len(board)
    
    # Corner control (weight: 25)
    corners = [(0,0), (0,size-1), (size-1,0), (size-1,size-1)]
    for x,y in corners:
        if board[x,y] == player:
            score += 25
        elif board[x,y] == opponent:
            score -= 25
            
    # Edge stability (weight: 5)
    for i in range(size):
        if board[0,i] == player: score += 5
        if board[size-1,i] == player: score += 5
        if board[i,0] == player: score += 5
        if board[i,size-1] == player: score += 5
        
        if board[0,i] == opponent: score -= 5
        if board[size-1,i] == opponent: score -= 5
        if board[i,0] == opponent: score -= 5
        if board[i,size-1] == opponent: score -= 5

    # Mobility (weight: 3)
    player_moves = len(get_valid_moves(board, player))
    opponent_moves = len(get_valid_moves(board, opponent))
    score += 3 * (player_moves - opponent_moves)
    
    # Piece differential (weight: 1)
    score += np.sum(board == player) - np.sum(board == opponent)
    
    return score

  def order_moves(self, board, moves, player):
    """Sort moves by immediate board evaluation for better alpha-beta pruning"""
    scored_moves = []
    for move in moves:
        temp_board = deepcopy(board)
        execute_move(temp_board, move, player)
        score = self.evaluate_board(temp_board, player, 3-player)
        scored_moves.append((score, move))
    return [m for s,m in sorted(scored_moves, reverse=True)]

  def minimax(self, board, depth, is_max, player, opponent, alpha, beta, start_time, time_limit):
    """Minimax algorithm with alpha-beta pruning and time limit checking"""
    # Check for timeout
    if time.time() - start_time > time_limit:
        raise TimeoutError()
        
    # Base cases
    if depth == 0 or check_endgame(board, player, opponent)[0]:
        return self.evaluate_board(board, player, opponent)
        
    current = player if is_max else opponent
    moves = get_valid_moves(board, current)
    
    # If no moves available, pass turn
    if not moves:
        return self.minimax(board, depth-1, not is_max, player, opponent, 
                          alpha, beta, start_time, time_limit)
    
    # Initialize value based on player
    value = float('-inf') if is_max else float('inf')

    # Try each move and update alpha/beta bounds
    for move in moves:
        new_board = deepcopy(board)
        execute_move(new_board, move, current)
        
        eval = self.minimax(new_board, depth-1, not is_max, player, opponent, 
                          alpha, beta, start_time, time_limit)
        
        if is_max:
            value = max(value, eval)
            alpha = max(alpha, eval)
        else:
            value = min(value, eval)
            beta = min(beta, eval)
        
        # Prune remaining branches if possible
        if beta <= alpha:
            break
            
    return value

  def iterative_deepening(self, board, player, opponent, depth, start_time, time_limit):
    """
        Iterative deepening search:
        - Starts with ordered moves for better pruning
        - Returns best move found within time limit
        - Handles timeouts gracefully
    """

    if time.time() - start_time > time_limit:
        raise TimeoutError()
        
    best_move = None
    best_score = float('-inf')
    moves = get_valid_moves(board, player)
    
    if not moves:
        return None, 0
        
    # Order moves for better pruning
    ordered_moves = self.order_moves(board, moves, player)
    
    for move in ordered_moves:
        if time.time() - start_time > time_limit:
            raise TimeoutError()
            
        new_board = deepcopy(board)
        execute_move(new_board, move, player)
        
        try:
            score = self.minimax(new_board, depth-1, False, 
                               player, opponent,
                               float('-inf'), float('inf'),
                               start_time, time_limit)
                               
            if score > best_score:
                best_score = score
                best_move = move
                
        except TimeoutError:
            if best_move is not None:
                return best_move, best_score
            return moves[0], 0
            
    return best_move, best_score
