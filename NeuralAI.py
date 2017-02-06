import tensorflow as tf
import os

class NeuralAI:
    def __init__(self, engine, player, search_depth, model_path, *args):
        self.engine = engine
        self.player = player
        self.model_path = model_path
        self.from_graph = self.load_graph('\\move_from-meta')
        self.to_graph = self.load_graph('\\move_to-meta')


    # def restore_vars(self):
    #     saver.restore(sess, 'results/model.ckpt.data-1000-00000-of-00001')
    

    def get_move(self, moves):
        number_of_moves = len(moves)
        c = random.randrange(0, number_of_moves)
        return moves[c]


    def load_graph(self, from_path):
        frozen_graph_filename = os.path.abspath(self.model_path + from_path)

        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we can use again a convenient built-in function to import a graph_def into the
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )
        print('loaded')
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