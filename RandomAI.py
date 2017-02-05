#currently just randomly selects a move
#next up: rate moves based on tree
import random

ai_type = 0

class RandomAI:
    def __init__(self, player, engine, *args):
        self.engine = engine
        self.player = player
    

    def get_move(self, moves):
        number_of_moves = len(moves)
        c = random.randrange(0, number_of_moves)
        return moves[c]