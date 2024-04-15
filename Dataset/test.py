import chess.engine


def main():
    stockfish_path = "./stockfish/src/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.quit()



if __name__ == "__main__":
    main()