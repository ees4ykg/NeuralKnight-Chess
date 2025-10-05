import chess.pgn
import numpy as np
import os
import chess
import chess.pgn as pgn
import pickle
import time


piece_types = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

def board_state_to_array(board):
    ''' convert a chess board object to a 14x8x8 numpy array
    the first 12 channels represent the positions of each peice type on the 8x8 chess board
    the 13th channel shows all legal moves on the board
    the 14th channel shows whos turn it is to move
    '''
    # create the array
    board_array = np.zeros((14, 8, 8))

    
    # get the board state
    board_state = board.fen().split()[0]  # Only take the board position part

    # get the board rows
    board_rows = board_state.split("/")  


    #iterate through the board rows
    row_index = 0
    for row in board_rows:
        col_index = 0
        for char in row:

            if char in piece_types.keys():
                board_array[piece_types[char], row_index, col_index] = 1
        
            
            # else if the char is a number,
            elif char.isdigit():
                col_index += int(char) - 1
            col_index += 1
        row_index += 1

    # get the legal moves
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        row_to = 8 - row_to
        board_array[12, row_to - 1, col_to - 1] = 1

    # get the turn
    turn = board.turn

    #make the 14th channel all 1s if the turn is white or 0s if the turn is black
    board_array[13, :, :] = 1 if turn == chess.WHITE else 0
    return board_array


def read_pgn_data(data_folder, limit_of_games=None, file_names=None):
    '''
    Loads the pgn data from the data folder and returns a list of games.
    Args:
        data_folder: The path to the data folder.
        limit_of_games: The maximum number of games to load.
        file_names: The names of specific files to load otherwise all files will be loaded.
    Returns:
        A list of games.
    '''
    games = []

    files_to_load = file_names if file_names is not None else os.listdir(data_folder)

    for file in files_to_load:
        if file.endswith(".pgn"):

            with open(os.path.join(data_folder, file), 'r') as pgn_file:

                counter = 0
                while True:
                    if limit_of_games is not None and len(games) >= limit_of_games:
                        break
                    game = pgn.read_game(pgn_file)

                    if game is None:
                        break
                    games.append(game)


                    counter += 1
                    if counter % 10000 == 0:
                        print(f"Processed {counter} games")
    return games

def save_input_for_nn(games, output_folder=None, mapping_name='model_move_to_idx'):
    '''
    Creates the input for the neural network and saves it to .npy file.
    Args:
        games: A list of games.
        output_file: The path to write the output file to.
    Returns:
        A list of inputs and a list of outputs, and the move mapping.
    '''

    x_data = []
    y_data_uci = []
    

    print(f"Converting {len(games)} games")

    start_time = time.time()
    counter = 0
    for game in games:

        board = game.board()

        for move in game.mainline_moves():
            x_data.append(board_state_to_array(board))
            y_data_uci.append(move.uci())
            board.push(move)


        counter += 1
        if counter % 10000 == 0:
            print(f"Converted {counter} games")

    end_time = time.time()

    print(f"Converted {len(games)} games in {end_time - start_time} seconds")


    # Create a mapping from UCI moves to integers
    unique_moves = sorted(set(y_data_uci))  # Get all unique moves
    move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
    #idx_to_move = {idx: move for move, idx in move_to_idx.items()}
    
    # Convert UCI strings to integers
    y_data = [move_to_idx[move] for move in y_data_uci]
    

    #save the data to .npy files
    x_path = os.path.join(output_folder, "x_data.npy")
    y_path = os.path.join(output_folder, "y_data.npy")
    
    #save the data to .npy files
    np.save(x_path, np.array(x_data))
    np.save(y_path, np.array(y_data))
    
    #save the move to idx mapping to a pickle file
    with open(f"models/{mapping_name}", "wb") as file:
        pickle.dump(move_to_idx, file)


    return x_data, y_data, move_to_idx


if __name__ == "__main__":
    print("Reading data...")
    games = read_pgn_data("raw_training_data", limit_of_games=10000*20)
    print("Data read successfully")
    print("Saving data...")
    save_input_for_nn(games, output_folder="cleaned_data", mapping_name='model2_heavy_move_to_idx')
    print("Data saved successfully")