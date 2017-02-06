
import Renderer as r
import GameEngine as g
import Human as h
import RandomAI
import time
import os
import argparse

FIRST_AI = RandomAI.RandomAI #RandomAI, MonteCarloAI, MinimaxAI
SECOND_AI = RandomAI.RandomAI

MODEL_PATH = os.path.abspath(os.getcwd() + '\\models\\predictive-meta')

# SECOND_AI = NeuralAI.NeuralAI

# humans = 0, 1, 2
def play_game(engine, humans = 1, renderer = None, AI1 = None, AI2 = None, search_depth=4):
    engine.board_setup()

    players = []
    if humans == 0:
        players.append(AI1(engine, 0, search_depth, MODEL_PATH))
        players.append(AI2(engine, 1, search_depth, MODEL_PATH))
    elif humans == 1:
        players.append(h.Human(engine, 0, renderer))
        players.append(AI2(engine, 1, search_depth, MODEL_PATH))
    elif humans == 2:
        players.append(h.Human(engine, 0, renderer))
        players.append(h.Human(engine, 1, renderer))
    else:
        raise Exception('This is a two player game, you listed more than 2 humans, or less than 0.')

    if renderer != None:
        renderer.draw_board()

    turn = 0
    moves_this_game = 0

    while True:
        moves = engine.all_legal_moves(turn)
        winner = engine.check_winner(turn, moves)
        if winner != 3:
            break

        move_choice = players[turn].get_move(moves)
        engine.move(move_choice)

        if renderer != None:
            renderer.draw_board()

        moves_this_game += 1
        turn = 1 - turn 
    return winner, moves_this_game



def game_start(args):
    engine = g.GameEngine(int(args.size))
    re = None
    num_games = int(args.number)
    num_humans = int(args.humans)
    board_size = int(args.size)
    gui = int(args.graphical)

    if gui:
        re = r.Renderer(engine)
        re.window_setup(500, 500)


    for i in range(num_games):
        winner, num_moves = play_game(engine, num_humans, re, FIRST_AI, SECOND_AI, board_size)
        print('Player', i, 'won in', num_moves, 'moves.')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graphical', default=1, help='whether or not to show the gui')
    parser.add_argument('number', default=1, help='How many games to play')
    parser.add_argument('humans', default=0, help='2 is two player, 1 is vs AI, 0 is both AI')
    parser.add_argument('size', default=10, help='How big the board is')
    args = parser.parse_args()

    game_start(args)

    
if __name__ == '__main__':
    main()
