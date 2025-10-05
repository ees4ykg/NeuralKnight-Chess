from chess import Board, pgn
from data_preprocessing import board_state_to_array
import torch
from torch_model import NeuralKnightChessModel
import pickle
import numpy as np


def input_to_tensor(board: Board):
    matrix = board_state_to_array(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor


#load the model and the move to idx mapping
import pickle
with open("models/model1_heavy_move_to_idx", "rb") as file:
    move_to_int = pickle.load(file)
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load the model
model = NeuralKnightChessModel(num_classes=len(move_to_int))
model.load_state_dict(torch.load("models/10_epochs_model.pth"))
model.to(device)
model.eval()  # Set the model to evaluation mode (it may be reductant)


int_to_move = {v: k for k, v in move_to_int.items()}


# Function to make predictions
def predict_move(board: Board):
    '''
    Predicts the best move for the given board.
    Args:
        board: The board to predict the move for.
    Returns:
        The best move for the given board in UCI format.
    '''
    X_tensor = input_to_tensor(board).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
    
    logits = logits.squeeze(0)  # Remove batch dimension
    
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    
    return None


# Test the function

if __name__ == "__main__":
    board = Board()
    while True:
        print(board)
        board.push_san(input("Enter a move: "))
        print(board)
        board.push_san(predict_move(board))
        








