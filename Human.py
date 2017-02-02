class Human:
    def __init__(self, engine, side, gui = False, renderer = None):
        self.side = side
        self.gui = gui
        self.renderer = renderer
        self.engine = engine


    def get_move(self):
        # Implement console only functionality, need parser from command line.
        all_moves = self.engine.moves
        if self.gui == False:
            return False
        else:
            while True:
                coord1 = self.renderer.get_mouse_square()

                pos_before = []
                pos_after = []
                for x in range(1, len(all_moves), 4):
                    pos_before.append((all_moves[x], all_moves[x+1]))
                    pos_after.append((all_moves[x+2], all_moves[x+3]))

                if not coord1 in pos_before:
                    print('Piece cant move 1')
                    continue

                # if coord1[0] < 0 or coord1[0] > 9 or coord1[1] < 0 or coord1[1] > 9:
                #     print('Point selected is out of bounds, select again.')
                #     continue

                # moves = self.engine.legal_moves_for_piece(coord1, self.side)
                # print(moves, 'moves')


                # if len(moves) == 0:
                #     print('You cant move that')
                #     continue

                # populate arr with tuples of posible moves of the piece
                # self.renderer.disp_pos_moves(moves)

                coord2 = self.renderer.get_mouse_square()
                if not coord2 in pos_after:
                    print('Piece cannot move to that square')
                    continue

                k = 1
                counter = 0
                while(all_moves[k] != 0):
                    if all_moves[k] == coord1[0] and all_moves[k+1] == coord1[1] and all_moves[k+2] == coord2[0] and all_moves[k+3] == coord2[1]:
                        return counter
                    counter += 1
                    k += 4

                    if counter > 10000:
                        return 'gg'
                return 'gg'