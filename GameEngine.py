import math
import random
import inspect
import numpy
from copy import deepcopy
import time


class GameEngine:
    def __init__(self, n):
        self.size = n
        self.board = None
        self.owner = None
        self.visible = None
        self.movement = None
        self.move_history = []


    def convert_to_numpy_1D(self, b, b2):
        for x in range(self.size):
            for y in range(self.size):
                b2[y + (self.size * x)] = b[x][y]


    def get_2D_array(self, b):
        board_temp = [[0 for x in range(0, self.size)] for x in range(0, self.size)]
        for x in range(self.size):
            for y in range(self.size):
                board_temp[x][y] = b[x + (self.size * y)]
        return board_temp


    def board_setup(self):
        self.board = numpy.zeros((self.size * self.size,), dtype=numpy.int8)
        self.owner = numpy.zeros((self.size * self.size,), dtype=numpy.int8)
        self.visible = numpy.zeros((self.size * self.size,), dtype=numpy.int8)
        self.movement = numpy.zeros((self.size * self.size,), dtype=numpy.int8)

        owner_temp = [[2 for x in range(0, self.size)] for x in range(0, self.size)]
        visibility_temp = [[0 for x in range(0, self.size)] for x in range(0, self.size)]
        board_temp = [[0 for x in range(0, self.size)] for x in range(0, self.size)]

        # Rivers
        if self.size == 10:
            board_temp[4][2] = 13
            board_temp[5][2] = 13
            board_temp[4][3] = 13
            board_temp[5][3] = 13

            board_temp[4][6] = 13
            board_temp[5][6] = 13
            board_temp[4][7] = 13
            board_temp[5][7] = 13
        elif self.size == 6:
            board_temp[2][2] = 13
            board_temp[3][2] = 13

        for i in range(0, 2):
            if self.size == 10:
                rows = 4
                starting_pieces = [[12, 'Flag F', 1], [10, 'Bomb B', 6], [11, 'Spy Y', 1], [9, 'Scout S', 8], [8, 'Miner R', 5], [7, 'Sergeant T', 4], [6, 'Lieutenent L', 4], [5, 'Captain C', 4], [4, 'Major J', 3], [3, 'Colonel O', 2], [2, 'General G', 1], [1, 'Marshall M', 1]]
            elif self.size == 6:
                rows = 2
                starting_pieces = [[12, 'Flag F', 1], [10, 'Bomb B', 1], [11, 'Spy Y', 1], [9, 'Scout S', 1], [8, 'Miner R', 1], [7, 'Sergeant T', 1], [6, 'Lieutenent L', 1], [5, 'Captain C', 1], [4, 'Major J', 1], [3, 'Colonel O', 1], [2, 'General G', 1], [1, 'Marshall M', 1]]


            starting_locations = []
            randy = random.randrange(0, self.size-1)
            randy2 = random.randrange(0, self.size-1)

            for x in range(0, self.size):
                for y in range(0 + i*(self.size-rows), rows + i*(self.size-rows)):
                    if y == 0 and x == randy and i == 0:
                        owner_temp[y][x] = 0
                        board_temp[y][x] = 12
                        starting_pieces.pop(0)
                        continue
                    if y == rows + i*(self.size-rows)-1 and x == randy2 and i == 1:
                        owner_temp[y][x] = 1
                        board_temp[y][x] = 12
                        starting_pieces.pop(0)
                        continue
                    starting_locations.append((x, y, i))

            while len(starting_pieces) != 0:
                r1 = int(random.random()*(len(starting_pieces)))
                r2 = int(random.random()*(len(starting_locations)))

                owner_temp[starting_locations[r2][1]][starting_locations[r2][0]] = starting_locations[r2][2]
                board_temp[starting_locations[r2][1]][starting_locations[r2][0]] = starting_pieces[r1][0]

                starting_locations.pop(r2)
                starting_pieces[r1][2] -= 1
                if starting_pieces[r1][2] == 0:
                    starting_pieces.pop(r1)
        self.convert_to_numpy_1D(board_temp, self.board)
        self.convert_to_numpy_1D(owner_temp, self.owner)
        self.convert_to_numpy_1D(visibility_temp, self.visible)



    def print_board(self):
        board_to_print = self.get_2D_array(self.board)
        # owner_board = self.get_2D_array(self.owner)
        print()
        for x in range(self.size):
            arr_temp = []
            for y in range(self.size):
                if board_to_print[x][y]:
                    val = board_to_print[x][y]
                    if val == 12:
                        name = 'F'
                    elif val == 10:
                        name = 'B'
                    elif val == 11:
                        name = 'S'
                    elif val == -1:
                        name = 'L'
                    else:
                        name = val
                    arr_temp.append(name)
                else:
                    arr_temp.append(0)
            print(' '.join(map(str, arr_temp)))
        print()



    # Takes in 2 players and returns 0 for p1 winning and 1 for p2 winning, 2 for tie.
    # Also reveals.
    def battle(self, v1, v2):
        if v2 == 0:
            return 0
        if v2 == 10:
            if v1 == 8:
                return 0
            return 1
        if v1 == 11:
            if v2 == 1 or v2 == 12:
                return 0
            return 1
        if v2 == 11:
            if v1 == 1:
                return 1
            return 0
        if v1 < v2:
            return 0
        if v1 > v2:
            return 1
        return 2


    # Takes in coord1 [x1, y1] and coord2 [x2, y2]
    # This assumes check_legal has been run.
    # The return is whether or not battle() was run
    def move(self, move_taken):
        coord0 = move_taken[0]
        coord1 = move_taken[1]

        x1 = coord0[0] -1
        y1 = coord0[1] -1
        x2 = coord1[0] -1
        y2 = coord1[1] -1
        size = self.size


        v1 = self.board[x1 + size*y1]
        self.board[x1 + size*(y1)] = 0

        v2 = self.board[x2 + size*y2]

        if v2 == 0:
            self.board[x2 + size*y2] = v1
            self.visible[x2 + size*y2] = self.visible[x1 + size*y1]
            self.owner[x2 + size*y2] = self.owner[x1 + size*y1]
            self.movement[x2 + size*(y2)] = self.movement[x1 + size*(y1)] + 1
        else:
            winner = self.battle(v1, v2)

            if winner == 0:
                self.board[x2 + size*y2] = v1
                self.visible[x2 + size*y2] = 1
                self.owner[x2 + size*y2] = self.owner[x1 + size*y1]
                self.movement[x2 + size*(y2)] = self.movement[x1 + size*(y1)] + 1
            elif winner == 1:
                self.visible[x2 + size*y2] = 1
            elif winner == 2:
                self.board[x2 + size*y2] = 0
                self.owner[x2 + size*y2] = 2
                self.visible[x2 + size*y2] = 0
                self.movement[x2 + size*(y2)] = 0
        self.visible[x1 + size*y1] = 0
        self.owner[x1 + size*(y1)] = 2
        self.movement[x1 + size*(y1)] = 0


    def legal_square_for_piece(self, val, old_x, old_y, new_x, new_y, player, moves, speed):
        new_board_pos = new_x + self.size*new_y

        if self.board[new_board_pos] == 0:
            moves.append(((old_x + 1, old_y + 1), (new_x + 1, new_y + 1)))
            return 0

        elif self.board[new_board_pos] < 13:
            if player != self.owner[new_board_pos]:
                moves.append(((old_x + 1, old_y + 1), (new_x + 1, new_y + 1)))
        return 1


    def legal_moves_for_piece(self, val, x, y, player, moves):
        if player != self.owner[x + self.size*y]: # handles if board == 0 and board == 13
            return
        elif val == 10 or val == 12:
            return

        speed = 2
        if val == 9:
            speed = self.size+1

        for p in range(1, speed):
            new_x = x+p
            if new_x < 0  or new_x > self.size-1:
                break

            if self.legal_square_for_piece(val, x, y, new_x, y, player, moves, speed):
                break

        for p in range(1, speed):
            new_x = x-p
            if new_x < 0  or new_x > self.size-1:
                break

            if self.legal_square_for_piece(val, x, y, new_x, y, player, moves, speed):
                break

        for p in range(1, speed):
            new_y = y+p
            if new_y < 0  or new_y > self.size-1:
                break

            if self.legal_square_for_piece(val, x, y, x, new_y, player, moves, speed):
                break

        for p in range(1, speed):
            new_y = y-p
            if new_y < 0  or new_y > self.size-1:
                break

            if self.legal_square_for_piece(val, x, y, x, new_y, player, moves, speed):
                break


    def all_legal_moves(self, player):
        moves = []
        for i in range(self.size*self.size):
            c_x = int(i % self.size)
            c_y = int(i / self.size)
            val = self.board[c_x + self.size*c_y]
            self.legal_moves_for_piece(val, c_x, c_y, player, moves)
        return moves


    # Takes in player to see if stalemate or something, might be able to removed
    # Returns True or False for if game is over, second is for resultx
    def check_winner(self, player, moves):
        flag0 = False
        flag1 = False
        for i in range(self.size*self.size):
            if self.board[i] == 12:
                if self.owner[i] == 0:
                    flag0 = True
                elif self.owner[i] == 1:
                    flag1 = True
                if flag0 and flag1:
                    break
        if not flag0:
            return 1
        elif not flag1:
            return 0

        if len(moves) == 0:
            opposite_moves = self.all_legal_moves(1-player)
            if len(opposite_moves) == 0:
                return 2
            return 1-player
        return 3




    # Code for minimax

    # def move_track(self, coord1, coord2):
    #     p1 = self.board[coord1[0]][coord1[1]]
    #     self.board[coord1[0]][coord1[1]] = 0
    #     if not isinstance(self.board[coord2[0]][coord2[1]], Piece):
    #         self.board[coord2[0]][coord2[1]] = p1
    #         return p1, 0

    #     p2 = self.board[coord2[0]][coord2[1]]
    #     winner = self.battle(p1, p2)

    #     if winner == 0:
    #         self.board[coord2[0]][coord2[1]] = p1
    #     elif winner == 2:
    #         self.board[coord2[0]][coord2[1]] = 0

    #     return p1, p2


    # # Functions for minimax
    # def push_move(self, coord1, coord2):
    #     p1, p2 = self.move_track(coord1, coord2)
    #     self.move_history.append((coord1, coord2, p1, p2))


    # def pop_move(self):
    #     c = self.move_history.pop()
    #     self.board[c[0][0]][c[0][1]] = c[2]
    #     self.board[c[1][0]][c[1][1]] = c[3]

