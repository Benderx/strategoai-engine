import tensorflow as tf
import os
import random

class NeuralAI:
    def __init__(self, engine, player, search_depth, model_path=None, *args):
        self.engine = engine
        self.player = player
        self.model_path = model_path

        self.sess = tf.Session()
        self.from_graph = self.load_graph('move_from')
        self.to_graph = self.load_graph('move_to')



    # def restore_vars(self):
    #     saver.restore(sess, 'results/model.ckpt.data-1000-00000-of-00001')
    

    def get_move(self, moves):
        # all_vars = [n.name for n in tf.get_default_graph().as_graph_def().node]
        all_vars2 = [v for v in tf.all_variables()]
        op = self.from_graph.get_operations()
        # print([m.values().name for m in op])
        # print([c.name for c in tf.get_collection(tf.GraphKeys.VARIABLES, scope=str('move_from/readout'))])
        
        # for x in all_vars:
        #     # print(x[0:17])
        #     if x[0:17] == 'move_from/readout':
        #         print(x)
        # last_layer_var = [v for v in tf.all_variables() if v.name == "move_from/readout/add"][0]
        # all_vars = tf.get_collection('vars')
        # print(all_vars)
        # for v in all_vars2:
        #     print(v.name)
        #     v_ = self.sess.run(v)
        #     print(v_)

        cross = None
        for o in op:
            if o.name == 'move_from/SoftmaxCrossEntropyWithLogits':
                print(o.name)
                corss = o

        o.run(feed_dict={owner_t: self.engine.owner, board_t: self.engine.board, keep_prob: 0})

        number_of_moves = len(moves)
        c = random.randrange(0, number_of_moves)
        return moves[c]


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