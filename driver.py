
import Renderer as r
import GameEngine as g
import Human as h
import RandomAI
import time
import sqlite3
import os
import argparse
import threading
import pandas

FIRST_AI = RandomAI.RandomAI #RandomAI, MonteCarloAI, MinimaxAI
SECOND_AI = RandomAI.RandomAI


# humans = 0, 1, 2
def play_game(engine, humans = 1, db_stuff = None, gui = False, renderer = None, AI1 = None, AI2 = None):
    engine.board_setup()
    tracking = True
    if db_stuff == None:
        tracking = False
    # engine.print_board()
    players = []
    if humans == 0:
        players.append(AI1(0, engine, 4))
        players.append(AI2(1, engine, 4))
    elif humans == 1:
        players.append(h.Human(engine, 0, gui, renderer))
        players.append(AI2(1, engine, 4))
    elif humans == 2:
        players.append(h.Human(engine, 0, gui, renderer))
        players.append(h.Human(engine, 1, gui, renderer))
    else:
        raise Exception('This is a two player game, you listed more than 2 humans, or less than 0.')

    if gui:
        renderer.draw_board()

    state_tracker = []
    turn = 0
    moves_this_game = 0

    while True:
        engine.all_legal_moves(turn)

        winner = engine.check_winner(turn)

        if winner != 0:
            break

        move = players[turn].get_move()
        engine.move(move)
        
        if gui:
            renderer.draw_board()
        
        turn = 1 - turn 
        timing_samples += 1 
    return winner



def game_start(args):
    engine = g.GameEngine(int(args.size))
    re = None
    num_games = int(args.number)


    for i in range(num_games):
        winner = play_game(engine, int(args.humans), FIRST_AI, SECOND_AI, int(args.size))
        print('game ', i, ': ', results[0], ' won in', results[1], 'moves', 'MP_PC:', float(results[1])/time)




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
