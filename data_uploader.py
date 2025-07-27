import chess.pgn
import pandas as pd
import boto3
import time
import os

dynamodb = boto3.resource('dynamodb', region_name='eu-west-2')

table = dynamodb.Table('NeuralKnight-Chess_database')

def upload_single_entry(item):
    try:
        table.put_item(Item=item)
    except Exception as e:
        print(f"Error uploading item: {e}")
        return False
    return True

def upload_pgn_file(pgn_file_path):
    """
    Uploads all game data from a pgn file to the dynamodb database.  
    Args:
        pgn_file_path: The path to the pgn file to upload.
    Returns:
        True if the upload was successful, False otherwise.
    """
    try:
        pgn = open(pgn_file_path)
        with table.batch_writer() as batch:
            
            start_time = time.time()
            counter = 0
            move_data_array = []

            while True: 
                game = chess.pgn.read_game(pgn) # read the next game from the pgn file  
                result = game.headers["Result"]
                board = game.board()

                if game is None:
                    break

                for move in game.mainline_moves(): # iterate through the moves in the game
                    FEN = board.fen() # get the FEN of the board
                    best_move = move.uci() # get the next move played

                    Item={
                        "FEN": FEN,
                        "best_move": best_move,
                        "game_outcome": result
                    }
                    #print(Item)
                    move_data_array.append(Item)
                    board.push(move)
                counter += 1
                print(f"Uploaded {counter} games from {pgn_file_path} in {time.time() - start_time} seconds")

            for Item in move_data_array:
                batch.put_item(Item)

        pgn.close()
        return True
    
    except Exception as e:
        print(f"Error uploading pgn file {pgn_file_path}: {e}")
        return False
    
            
            



if __name__ == "__main__":
    for file in os.listdir("training_data"):
        if file.endswith(".pgn"):
            print(f"Uploading {file}")
            assert upload_pgn_file(f"training_data/{file}")
    print("All files uploaded successfully")