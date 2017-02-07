import tensorflow as tf
import os
import random

class NeuralAI:
    def __init__(self, engine, player, search_depth, model_path=None, *args):
        self.engine = engine
        self.player = player
        self.model_path = model_path

        self.sess = tf.Session()
        self.from_graph, saver0, total_path0, model0 = self.load_graph('move_from')
        self.to_graph, saver1, total_path1, model1 = self.load_graph('move_to')
        self.sess.run(tf.global_variables_initializer())

        self.load_weights(total_path0, model0, saver0)
        self.load_weights(total_path1, model1, saver1)



    # def restore_vars(self):
    #     saver.restore(sess, 'results/model.ckpt.data-1000-00000-of-00001')
    

    def get_move(self, moves):
        number_of_moves = len(moves)
        c = random.randrange(0, number_of_moves)
        return moves[c]


    def load_weights(self, total_path, path, saver):
        dirs = os.listdir(total_path)
        found = False
        for x in dirs:
            if os.path.isfile(os.path.join(total_path, x)):
                if x[len(path):len(path)+5] == '.data':
                    found = True
                    break

        if not found:
            raise Exception('Could not find weight file in', total_path)

        print('NeuralAI - Loading weights for:', path)

        coll = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(path))
        print([v.name for v in coll])

        with tf.variable_scope(str(path)):
            saver.restore(self.sess, tf.train.latest_checkpoint(total_path, ))

        print('NeuralAI - Loaded weights for:', path)


    def load_graph(self, path):
        curr_dir = os.path.join(os.getcwd(), 'models', 'curr')
        if not os.path.isdir(curr_dir):
            raise Exception('Directory where models should be stored does not exist, please create directory', curr_dir, 'and place model saves in there.')

        total_path = os.path.join(curr_dir, path)
        if not os.path.isdir(total_path):
            raise Exception('Directory where', total_path, 'should be stored does not exist, please create directory', total_path, 'and put model there.')

        print('NeuralAI - Loading model:', path)


        meta_file = os.path.join(total_path, path + '.meta')
        saver = tf.train.import_meta_graph(meta_file)
        graph = tf.get_default_graph()


        print('NeuralAI - Loaded model:', path)

        return graph, saver, total_path, path


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