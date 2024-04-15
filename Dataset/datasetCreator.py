import chess.pgn
import chess.engine
import re
import pandas as pd
import numpy as np

def evaluateBoard(board, depth=15):
    # Path to the Stockfish executable
    stockfish_path = "./stockfish/src/stockfish"
    
    # Initialize the chess engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Evaluate the position using the chess engine
    info = engine.analyse(board, chess.engine.Limit(depth=depth))  # You can adjust the analysis time
    
    
    # Close the chess engine when done
    engine.quit()
    
    return centipawnToInt(str(info["score"].white()))


def centipawnToInt(cp):
    match = re.search(r'([+-]?\d+)', cp)

    if match:
        integer_score = int(match.group())
    
    return integer_score


def make_matrix(board): #type(board) == chess.Board()
    pgn = board.epd()
    foo = []  #Final board
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo.append(0)
            else:
                if thing == 'P':
                    foo.append(1)
                elif thing == 'p':
                    foo.append(-1)
                elif thing == 'N':
                    foo.append(2)
                elif thing == 'n':
                    foo.append(-2)
                elif thing == 'B':
                    foo.append(3)
                elif thing == 'b':
                    foo.append(-3)
                elif thing == 'R':
                    foo.append(4)
                elif thing == 'r':
                    foo.append(-4)
                elif thing == 'Q':
                    foo.append(5)
                elif thing == 'q':
                    foo.append(-5)
                elif thing == 'K':
                    foo.append(6)
                elif thing == 'k':
                    foo.append(-6)
                    
                else:
                    foo.append(thing)
    return foo

def pawnsArray(arr):
    return [1 if x == 1 else -1 if x == -1 else 0 for x in arr]

def knightsArray(arr):
    return [2 if x == 2 else -2 if x == -2 else 0 for x in arr]

def bishopsArray(arr):
    return [3 if x == 3 else -3 if x == -3 else 0 for x in arr]

def rooksArray(arr):
    return [4 if x == 4 else -4 if x == -4 else 0 for x in arr]

def queensArray(arr):
    return [5 if x == 5 else -5 if x == -5 else 0 for x in arr]

def kingsArray(arr):
    return [6 if x == 6 else -6 if x == -6 else 0 for x in arr]


def makeRandomMove(board):
    legal_moves = list(board.legal_moves)

    if len(legal_moves) == 0:
        print("The game is over.")
    else:
        # Choose a random move from the list
        import random
        random_move = random.choice(legal_moves)
    
        # Make the chosen move on the board
        board.push(random_move)




board = chess.Board()

res = []

nRows = 0
cont = 0

while nRows < 30000:
    board = chess.Board()
    nMoves = 0
    while not board.is_game_over() and not board.is_insufficient_material() and nMoves < 200:  # Changed the condition
        makeRandomMove(board)
        
        if board.turn:
            arr = make_matrix(board)
            res.append([arr, pawnsArray(arr), knightsArray(arr), bishopsArray(arr), rooksArray(arr), queensArray(arr), kingsArray(arr), evaluateBoard(board)])
            nRows = nRows + 1  # Counting the rows
        nMoves = nMoves + 1
        print("\n" + str(nRows) + "\\" +str(cont+1))
        print(board)
        
    cont = cont + 1
    

df = pd.DataFrame(res)

new_column_names = ['posiciones','pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings', 'y']
df.columns = new_column_names

df.to_json('data.json')
