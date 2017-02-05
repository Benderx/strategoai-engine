import tensorflow as tf



class NeuralAI:
    def __init__(self, player, engine, *args):
        self.engine = engine
        self.player = player
        self.board = engine.get_board()
        self.default_graph = self.get_graph()


    def get_graph(self):
        # Load the VGG-16 model in the default graph
        saver = tf.train.import_meta_graph('META FILE HERE')
        # Access the graph
        graph = tf.get_default_graph()
        return graph


    def restore_vars(self):
        saver.restore(sess, 'results/model.ckpt.data-1000-00000-of-00001')
    

    def get_move(self, all_moves):
        c = random.choice(all_moves)
        return c





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