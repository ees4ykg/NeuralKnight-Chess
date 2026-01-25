import time
import os
from dotenv import load_dotenv
import chess
import chess.engine
import random
import h5py
import numpy as np
from data_preprocessing import board_state_to_array

start = time.time()

load_dotenv()

# environment key:
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH")


#def is_valid_position(board):
#    
#    # Position must be legal
#    if not board.is_valid():
#        return False
#    else:
#        return True
#    # Using stockfish to verify a checkmate exists
#    info = engine.analyse(board, chess.engine.Limit(depth=15))
#    score = info["score"].white()
#    
#    # Check for mate
#    if score.is_mate():
 #       return True
 #   
 #   # Check for high evaluation (centipawns / 100 = pawns)
 #   centipawns = score.score()
 #   if centipawns is not None and abs(centipawns) > 500:  # > 5 pawns
##        return True
    
#    return False


def generate_endgame_board(white_pieces, black_pieces, winning_colour):
    """generalising the below function to generate any type of endgame position"""
    board = chess.Board()
    board.clear()

    squares = list(range(8,56))  # not include first or last row to avoid some illegal board states
    random.shuffle(squares)

    counter = 0
    for piece in white_pieces:
        board.set_piece_at(squares[counter], chess.Piece(piece, chess.WHITE))
        counter += 1
    
    for piece in black_pieces:
        board.set_piece_at(squares[counter], chess.Piece(piece, chess.BLACK))
        counter += 1
    
    board.turn = winning_colour

    return board



def white_generate_KQ_vs_K():
    """Generate a random K+Q vs K position"""
    board = chess.Board()
    board.clear()

    squares = list(range(64))  # 0-63
    random.shuffle(squares)

    white_king_sq = squares[0]
    white_queen_sq = squares[1]
    black_king_sq = squares[2]

    board.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(white_queen_sq, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))

    board.turn = chess.WHITE

    return board

def black_generate_KQ_vs_K():
    """Generate a random K+Q vs K position"""
    board = chess.Board()
    board.clear()

    squares = list(range(64))  # 0-63
    random.shuffle(squares)

    black_king_sq = squares[0]
    black_queen_sq = squares[1]
    white_king_sq = squares[2]

    board.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(black_queen_sq, chess.Piece(chess.QUEEN, chess.BLACK))
    board.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))

    board.turn = chess.BLACK

    return board


def generate_endgame_data(board, winning_colour,depth, num_turns=1000): # I am going to assume the winning colour starts

    x_data_boards = []
    y_data_ucis = []
    move_count = 0

    while not board.is_game_over():
        result = engine.play(board, chess.engine.Limit(depth=depth))
        best_move = result.move

        if board.turn == winning_colour:
            x_data_boards.append(board.copy())
            y_data_ucis.append(best_move.uci())
        
        board.push(best_move)
        print(board)
        move_count += 1
        print(move_count)
        print('===========================================================================================================================')

    return {'x_data': x_data_boards[-1*num_turns:], 'y_data': y_data_ucis[-1*num_turns:], 'checkmate': board.is_checkmate()}



def create_hdf5_file(filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset(
            'positions',
            shape=(0, 14, 8, 8),
            maxshape=(None, 14, 8, 8),
            dtype='float32'
        )
        f.create_dataset(
            'moves',
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype()
        )
    print(f"Created {filename}")


def append_to_hdf5(filename, x_batch, y_batch):
    """
    x_batch: list of numpy arrays, each shape (14, 8, 8)
    y_batch: list of strings, e.g. ["e2e4", "d7d5"]
    """
    with h5py.File(filename, 'a') as f:
        pos = f['positions']
        moves = f['moves']
        
        current_size = pos.shape[0]
        new_size = current_size + len(x_batch)
        
        pos.resize(new_size, axis=0)
        moves.resize(new_size, axis=0)
        
        pos[current_size:new_size] = np.array(x_batch)
        moves[current_size:new_size] = y_batch
    
    print(f"Added {len(x_batch)} positions. Total: {new_size}")
    return new_size


def generate_append_many_endgames(white_pieces, black_pieces, winning_colour,depth):
    x_batch = []
    y_batch = []

    for i in range(100): # 100 is the batch size
        while True:
            board = generate_endgame_board(white_pieces, black_pieces, winning_colour)

            if board.is_valid():
                game_data = generate_endgame_data(board, winning_colour, depth)
                break
        if game_data['checkmate']:
            x_board_arrays = [board_state_to_array(board) for board in game_data['x_data']]
            x_batch.extend(x_board_arrays)
            y_batch.extend(game_data['y_data'])

    global HDF5_file_size
    HDF5_file_size = append_to_hdf5("synthetic_data/stockfish_generated_data.h5", x_batch, y_batch)


if __name__ == "__main__":

    start = time.time()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    HDF5_file_size = 0

    ### generating games as for King+queen vs king
    #n=0
    #while HDF5_file_size < 250000:
    #    generate_append_many_endgames(white_pieces=[chess.KING, chess.PAWN], black_pieces=[chess.KING], winning_colour=chess.WHITE,depth=15)
    #    print(f"Batch {n+1} complete for K+P vs K white")
    #    n += 1

    #while HDF5_file_size < 500000:
    #    generate_append_many_endgames(white_pieces=[chess.KING], black_pieces=[chess.KING,chess.PAWN], winning_colour=chess.BLACK,depth=15)
    #    print(f"Batch {n+1} complete for K+P vs K black")
    #    n += 1


    ### generating games as for King+bishop+knight vs king
    #n=0
    #while HDF5_file_size < 700000:
    #    generate_append_many_endgames(white_pieces=[chess.KING, chess.BISHOP, chess.KNIGHT], black_pieces=[chess.KING], winning_colour=chess.WHITE,depth=15)
    #    print(f"Batch {n+1} complete for K+B+N vs K white")
    #    n += 1

    #while HDF5_file_size < 900000:
    #    generate_append_many_endgames(white_pieces=[chess.KING], black_pieces=[chess.KING, chess.BISHOP, chess.KNIGHT], winning_colour=chess.BLACK,depth=15)
    #    print(f"Batch {n+1} complete for K+B+N vs K black")
    #    n += 1


    ### generating games as for King+rook vs king
    #n=0
    #while HDF5_file_size < 1100000:
    #    generate_append_many_endgames(white_pieces=[chess.KING, chess.ROOK], black_pieces=[chess.KING], winning_colour=chess.WHITE,depth=15)
    #    print(f"Batch {n+1} complete for K+R vs K white")
    #    n += 1

    #while HDF5_file_size < 1300000:
    #    generate_append_many_endgames(white_pieces=[chess.KING], black_pieces=[chess.KING, chess.ROOK], winning_colour=chess.BLACK,depth=15)
    #    print(f"Batch {n+1} complete for K+R vs K black")
    #    n += 1


    ### combining the lichess elite player data
    import chess.pgn as pgn
    
    data_folder = "raw_training_data"
    pgn_files = [f for f in os.listdir(data_folder) if f.endswith(".pgn")]
    
    batch_size = 50  # games per batch (~3,000-4,000 positions)
    
    for pgn_filename in pgn_files:
        if HDF5_file_size >= 3000000:
            break
            
        filepath = os.path.join(data_folder, pgn_filename)
        print(f"Processing {pgn_filename}...")
        
        with open(filepath, 'r') as pgn_file:
            while HDF5_file_size < 3000000:
                x_batch = []
                y_batch = []
                games_in_batch = 0
                
                # Read a batch of games
                while games_in_batch < batch_size:
                    game = pgn.read_game(pgn_file)
                    if game is None:
                        break  # End of file
                    
                    board = game.board()
                    for move in game.mainline_moves():
                        x_batch.append(board_state_to_array(board))
                        y_batch.append(move.uci())
                        board.push(move)
                    
                    games_in_batch += 1
                
                if len(x_batch) == 0:
                    break  # No more games in this file
                
                # Append batch to HDF5
                HDF5_file_size = append_to_hdf5(
                    "synthetic_data/stockfish_generated_data.h5",
                    x_batch,
                    y_batch
                )
                print(f"Processed {games_in_batch} games, total positions: {HDF5_file_size}")
                
                if HDF5_file_size >= 3000000:
                    break
    
    print(f"Final dataset size: {HDF5_file_size} positions")



    
    engine.quit()
    
    end = time.time()
    print(f"Took {end - start:.2f} seconds")
    

