
import MonteCarloAI
import Renderer as r
import GameEngine as g
import Human as h
import MinimaxAI
import RandomAI
import time
import sqlite3
import os
import argparse
import threading
import c_bindings.engine_commands as c_bindings
import pandas

FIRST_AI = RandomAI.RandomAI #RandomAI, MonteCarloAI, MinimaxAI
SECOND_AI = MonteCarloAI.MonteCarloAI


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
    # else:
    #     engine.print_board()

    state_tracker = []
    turn = 0
    timing_total = 0
    timing_samples = 0
    moves_this_game = 0
    start = time.perf_counter()

    while True:
        # state_tracker.append(engine.get_compacted_board_state())
        
        # start = time.perf_counter()
        engine.all_legal_moves(turn)    # 5.4 x 10 ^ -6       10x10
        # end = time.perf_counter()

        winner = engine.check_winner(turn)     # 5.3 x 10 ^ -6       10x10

        
        if winner != 0:     # 1.4 x 10 ^ -7       10x10
            break
        

        move = players[turn].get_move()     # 5.0 x 10 ^ -6       10x10
        engine.move(move)     # 2.51 x 10 ^ -6       10x10

        
        moves_this_game += 1    #1.2 x 10 ^ -7       10x10
        moves_per_second += 1    #1.2 x 10 ^ -7       10x10
        

        if gui:
            renderer.draw_board()   # 1.4 x 10 ^ -7       10x10
        
        # else:
        #     engine.print_board()
        
        turn = 1 - turn    # 1.1 x 10 ^ -7       10x10
        timing_samples += 1   # 1.1 x 10 ^ -7       10x10
        # timing_total += end - start   # 1.1 x 10 ^ -7       10x10

    # print('avg:', timing_total/timing_samples)
    # time.sleep(.5)

    # end = time.perf_counter()
    # print(moves_this_game/(end-start))
    # exit()    


    if tracking:
        sql_game_insert =   """
                                INSERT INTO Game (WINNER)
                                VALUES (?);
                            """
        db_stuff[1].execute(sql_game_insert, str(winner))

        game_id = db_stuff[1].lastrowid
        state_tracker_packed = []
        for i in state_tracker:
            state_tracker_packed.append((str(game_id), str(i)))

        sql_state_insert =  """
                                INSERT INTO State (GAME_ID, BOARD)
                                VALUES (?, ?);
                            """
        db_stuff[1].executemany(sql_state_insert, state_tracker_packed)
        db_stuff[0].commit()
    return engine.check_winner(turn)


# Takes in database name and if you want to overwrite current, or add to it. Probably change in future for streamlined data creation
# Returns sqlite3 db connection
def init_db(dbpath = 'test.db', overwrite = True):
    exist = False
    if os.path.isfile(dbpath):
        exist = True

    if overwrite and exist:
        os.remove(dbpath)
    
    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()

    if overwrite or exist == False:
        sql_create_game =   """
                                CREATE TABLE Game (
                                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                                WINNER INTEGER NOT NULL);
                            """
        cursor.execute(sql_create_game)

        # How should we store the board?
        sql_create_state =  """
                                CREATE TABLE State (
                                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                                GAME_ID INTEGER NOT NULL,
                                BOARD CHAR(256) NOT NULL,
                                VISIBLE CHAR(256) NOT NULL,
                                OWNER CHAR(256) NOT NULL,
                                MOVEMENT CHAR(256) NOT NULL,
                                MOVE_MADE CHAR(32) NOT NULL,
                                FOREIGN KEY (GAME_ID) REFERENCES Game(ID));
                            """
        cursor.execute(sql_create_state)
    return [conn, cursor]


def print_moves_per_second(thread_name, delay, c):
    global moves_per_second;
    count = 0
    last = time.perf_counter()
    while count < c:
        count += 1
        while not time.perf_counter() - last > 1:
            time.sleep(.05)
        last = time.perf_counter()
        print('Moves per perf: ' + str(moves_per_second))
        moves_per_second = 0



def play_back_game(engine, results, renderer, board_size, db_stuff, game_iter):
    counter = engine.board_setup(results, board_size)

    turn = 0
    moves_this_game = results[1]

    board, visible, owner, movement, all_moves, move_taken = [], [], [], [], [], []
    while True:
        if renderer != None:
            renderer.draw_board()

        move = results[counter:counter+4]
        move_transformed = engine.move(move, board_size)
        if move_transformed == None:
            break
        if db_stuff != None:
            # board = engine.board.tostring()
            # visible = engine.visible.tostring()
            # owner = engine.owner.tostring()
            # movement = engine.movement.tostring()
            # print('here')
            # game_recorder.append([0, board, visible, owner, movement, move_transformed])
            board.append(engine.board)
            visible.append(engine.visible)
            owner.append(engine.owner)
            movement.append(engine.movement)
            move_taken.append(move_transformed)
            # all_moves.append(engine.all_legal_moves(turn))



        counter += 4
        turn = 1- turn
        if renderer != None:
            time.sleep(.5)

    df = pandas.DataFrame({'board':board,'visible': visible,
                           'owner': owner, 'movement': movement,'move_taken': move_taken})
    print(df)
    if db_stuff != None:
        df.to_csv('games.csv')

    #     try:
    #         sql_game_insert =   """
    #                                 INSERT INTO Game (WINNER)
    #                                 VALUES (?);
    #                             """
    #         db_stuff[1].execute(sql_game_insert, str(results[0]))
    #
    #         game_id = db_stuff[1].lastrowid
    #
    #         for i in game_recorder:
    #             i[0] = str(game_id)
    #             i = tuple(i)
    #
    #         sql_state_insert =  """
    #                                 INSERT INTO State (GAME_ID, BOARD, VISIBLE, OWNER, MOVEMENT, MOVE_MADE)
    #                                 VALUES (?, ?, ?, ?, ?, ?);
    #                             """
    #         db_stuff[1].executemany(sql_state_insert, game_recorder)
    #         db_stuff[0].commit()
    #     except:
    #         raise Exception("database insertion failed")
    #
    #     print("game", game_iter, "tracking over")
    # else:
    #     print("game", game_iter, "replay over")
    
    if renderer != None:
        time.sleep(1000)



# humans = 0, 1, 2
def play_c_game(engine, humans = 1, AI1 = None, AI2 = None, board_size = 10):

    if humans == 1:
        raise Exception("Humans cannot play during a c game")

    start = time.perf_counter()
    results = c_bindings.play_game(0, 1, 1000, board_size)
    end = time.perf_counter()

    return results, end-start


def game_start(args):
    engine = g.GameEngine(int(args.size))
    re = None
    if int(args.track):
        db_stuff = init_db('games.db', True)
    else:
        db_stuff = None
    num_games = int(args.number)


    for i in range(num_games):
        results, time = play_c_game(engine, int(args.humans), FIRST_AI, SECOND_AI, int(args.size))
        print('game ', i, ': ', results[0], ' won in', results[1], 'moves', 'MP_PC:', float(results[1])/time)

        if int(args.graphical) == 1 or int(args.track) == 1:
            if int(args.graphical) == 1:
                re = r.Renderer(engine)
                re.window_setup(500, 500)

            play_back_game(engine, results, re, int(args.size), db_stuff, i)

    if int(args.track):
        db_stuff[0].close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graphical', default=1, help='whether or not to show the gui')
    parser.add_argument('number', default=1, help='How many games to play')
    parser.add_argument('humans', default=0, help='2 is two player, 1 is vs AI, 0 is both AI')
    parser.add_argument('size', default=10, help='How big the board is')
    parser.add_argument('track', default=1, help='If database tracking happens')
    args = parser.parse_args()

    game_start(args)

    
if __name__ == '__main__':
    main()
