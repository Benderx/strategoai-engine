import tensorflow as tf
import os
import random
import numpy
import pandas
import math
import copy

class NeuralAI:
    def __init__(self, engine, player, search_depth, model_path=None, *args):
        self.engine = engine
        self.player = player
        self.model_path = model_path

        self.sess = tf.Session()
        self.from_graph = self.load_graph('move_from')
        self.to_graph = self.load_graph('move_to')


    def one_hot(self, val):
        temp = numpy.zeros(36, dtype = "int16")
        temp[val] = 1
        return temp


    def get_move(self, moves):
        possible_move_from = []
        for i in moves:
            f = i[0]
            total = (f[0]-1) + (f[1]-1)*6
            possible_move_from.append(total)

        # PULLING MOVE_FROM TENSORS

        board_place = None
        owner_place = None
        keep_prob_place = None
        move_from_place = None

        board_place = tf.get_default_graph().get_tensor_by_name("move_from/input/board_t:0")
        owner_place = tf.get_default_graph().get_tensor_by_name("move_from/input/owner_t:0")
        # move_from_place = tf.get_default_graph().get_tensor_by_name("move_from/input/move_from_one_hot_t:0")
        keep_prob_place = tf.get_default_graph().get_tensor_by_name("move_from/dropout/keep_prob_t:0")

        y_conv = None
        y_conv = tf.get_default_graph().get_tensor_by_name("move_from/readout/add:0")

        # EVAKL MOVE_FROM()
        result = list(y_conv.eval(feed_dict={board_place: [self.engine.board], owner_place: [self.engine.owner], keep_prob_place: 1.0}, session = self.sess)[0])

        print('move_from', result)
        # input()

        result_copy = list(copy.deepcopy(result))
        result_copy.sort(reverse=True)

        for i in result_copy:
            move_from_net = result.index(i)
            if move_from_net in possible_move_from:
                break

        print('net 1 done:', move_from_net)

        possible_move_to = []
        for i in moves:
            f = i[0]
            t = i[1]
            total = (f[0]-1) + (f[1]-1)*6
            if total == move_from_net:
                possible_move_to.append((t[0]-1) + (t[1]-1)*6)

        # PULLING MOVE_TO TENSORS

        board_place = None
        owner_place = None
        keep_prob_place = None
        move_to_place = None
        move_from_place = None

        board_place = tf.get_default_graph().get_tensor_by_name("move_to/input/board_t:0")
        owner_place = tf.get_default_graph().get_tensor_by_name("move_to/input/owner_t:0")
        move_from_place = tf.get_default_graph().get_tensor_by_name("move_to/input/move_from_one_hot_t:0")
        # move_to_place = tf.get_default_graph().get_tensor_by_name("move_to/input/move_to_one_hot_t:0")
        keep_prob_place = tf.get_default_graph().get_tensor_by_name("move_to/dropout/keep_prob_t:0")


        y_conv = None
        y_conv = tf.get_default_graph().get_tensor_by_name("move_to/readout/add:0")

        # EVAL MOVE_TO()
        result = list(y_conv.eval(feed_dict={board_place: [self.engine.board], owner_place: [self.engine.owner], move_from_place: [self.one_hot(move_from_net)], keep_prob_place: 1.0}, session = self.sess)[0])

        print('move_to', result)
        # input()

        result_copy = copy.deepcopy(result)
        result_copy.sort(reverse=True)

        for i in result_copy:
            move_to_net = result.index(i)
            if move_to_net in possible_move_to:
                break

        print('net 2 done:',move_to_net)


        move_from_net_2d = [0, 0]
        move_to_net_2d = [0, 0]

        move_from_net_2d[0] = (move_from_net % 6) + 1
        move_from_net_2d[1] = int(move_from_net / 6) + 1
        move_to_net_2d[0] = (move_to_net % 6) + 1
        move_to_net_2d[1] = int(move_to_net / 6) + 1

        return (tuple(move_from_net_2d), tuple(move_to_net_2d))


        # number_of_moves = len(moves)
        # c = random.randrange(0, number_of_moves)
        # return moves[c]


    def load_weights(self, total_path, model_name):
        dirs = os.listdir(total_path)
        found = False
        for x in dirs:
            if os.path.isfile(os.path.join(total_path, x)):
                if x[len(model_name):len(model_name)+5] == '.data':
                    found = True
                    break

        if not found:
            raise Exception('Could not find weight file in', total_path)

        print('NeuralAI - Loading weights for:', model_name)

        coll = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(model_name))

        new_saver = tf.train.Saver(coll)
        new_saver.restore(self.sess, tf.train.latest_checkpoint(total_path, ))

        print('NeuralAI - Loaded weights for:', model_name)


    def load_graph(self, model_name):
        curr_dir = os.path.join(os.getcwd(), 'models', 'curr')
        if not os.path.isdir(curr_dir):
            raise Exception('Directory where models should be stored does not exist, please create directory', curr_dir, 'and place model saves in there.')

        total_path = os.path.join(curr_dir, model_name)
        if not os.path.isdir(total_path):
            raise Exception('Directory where', total_path, 'should be stored does not exist, please create directory', total_path, 'and put model there.')

        print('NeuralAI - Loading model:', model_name)

        meta_file = os.path.join(total_path, model_name + '.meta')
        saver = tf.train.import_meta_graph(meta_file)
        graph = tf.get_default_graph()

        print('NeuralAI - Loaded model:', model_name)

        self.load_weights(total_path, model_name)

        return graph


# Restore

# sess = tf.Session()
# new_saver = tf.train.import_meta_graph('my-model.meta')
# new_saver.restore(sess, tf.train.latest_checkpoint('./'))
# all_vars = tf.get_collection('vars')
# for v in all_vars:
#     v_ = sess.run(v)
#     print(v_)


# Save

# w1 = tf.Variable(tf.truncated_normal(shape=[10]), name='w1')
# w2 = tf.Variable(tf.truncated_normal(shape=[20]), name='w2')
# tf.add_to_collection('vars', w1)
# tf.add_to_collection('vars', w2)
# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver.save(sess, 'my-model')
# # `save` method will call `export_meta_graph` implicitly.
# # you will get saved graph files:my-model.meta