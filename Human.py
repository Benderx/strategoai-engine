class Human:
    def __init__(self, engine, side, renderer = None):
        self.side = side
        self.renderer = renderer
        self.engine = engine


    def get_move(self, moves):
        # Implement console only functionality, need parser from command line.
        if self.renderer == None:
            raise Exception('humans cant play without gui')
            return False
        else:
            while True:
                coord1 = self.renderer.get_mouse_square()

                pos_before = []
                pos_after = []
                for x in moves:
                    pos_before.append((x[0][0], x[0][1]))
                    pos_after.append((x[1][0], x[1][1]))

                if not coord1 in pos_before:
                    print('Piece cant move')
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

                if (coord1, coord2) in moves:
                    return (coord1, coord2)
                else:
                    print('Piece cannot move to that square')
                    continue

                return 'gg'