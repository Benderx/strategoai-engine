import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import pandas
import numpy
import argparse
from tensorflow.python.framework import graph_util


CSV_LOCATION = 'games'
BATCH_SIZE = 100



def dir_location(name):
    i = 0
    while(os.path.isdir(os.path.abspath(name + '-v' + str(i)))):
        i += 1
    new_dir = name + '-v' + str(i)
    final_path = os.path.join(os.getcwd(), new_dir)
    os.mkdir(final_path)
    return final_path


def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = "Accuracy/predictions"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


# for now, just use full board state and predict piece chosen
def train_move_from(board, owner, move_from_one_hot, iterations):
    with tf.variable_scope('move_from'):
        with tf.name_scope('input'):
            board_t = tf.placeholder(tf.float32, [None, 36])
            owner_t = tf.placeholder(tf.float32, [None, 36])
            move_from_one_hot_t = tf.placeholder(tf.float32, [None, 36])
            shaped_board = tf.reshape(board_t, [-1,6,6])
            shaped_owner = tf.reshape(owner_t, [-1,6,6])
            shaped_state = tf.stack([shaped_board, shaped_owner], axis=-1)

        with tf.name_scope('layer_1'):
            W_conv1 = weight_variable([4, 4, 2, 32])
            b_conv1 = bias_variable([32])

            h_conv1 = tf.nn.relu(conv2d(shaped_state, W_conv1) + b_conv1)
            # h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('layer_2'):
            W_conv2 = weight_variable([4, 4, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
            # h_pool2 = max_pool_2x2(h_conv2)

        with tf.name_scope('connected_layer_1'):
            W_fc1 = weight_variable([6*6*64, 1024])
            b_fc1 = bias_variable([1024])
            h_conv2_flat = tf.reshape(h_conv2, [-1, 6*6*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        with tf.name_scope('connected_layer_2'):
            W_fc2 = weight_variable([1024, 1024])
            b_fc2 = bias_variable([1024])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        with tf.name_scope('connected_layer_3'):
            W_fc3 = weight_variable([1024, 1024])
            b_fc3 = bias_variable([1024])
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            h_fc2_drop = tf.nn.dropout(h_fc3, keep_prob)

        with tf.name_scope('readout'):
            W_r = weight_variable([1024, 36])
            b_r = bias_variable([36])

            y_conv = tf.matmul(h_fc2_drop, W_r) + b_r

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=move_from_one_hot_t))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(move_from_one_hot_t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # print([v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        samples = len(data)
        loc = 0
        for i in range(iterations):
            batch_board = board[loc:loc+BATCH_SIZE]
            batch_owner = owner[loc:loc+BATCH_SIZE]
            batch_move_from_one_hot = move_from_one_hot[loc:loc+BATCH_SIZE]

            loc = (loc + BATCH_SIZE) % samples
            if (i-1) % 1000 == 0 or i < 10:
                train_accuracy = accuracy.eval(feed_dict={
                    board_t: batch_board, owner_t: batch_owner, move_from_one_hot_t: batch_move_from_one_hot, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))

            train_step.run(feed_dict={board_t: batch_board, owner_t: batch_owner, move_from_one_hot_t: batch_move_from_one_hot, keep_prob: 0.5})
        saver = tf.train.Saver()

        model_name = 'move_from'
        dir_save = dir_location(model_name)
        saver.save(sess, os.path.join(dir_save, model_name))
        # freeze_graph(dir_save)

        print('Saved move_to model to:', dir_save)

def train_move_to(board, owner, move_from, move_to_one_hot, iterations):
    with tf.variable_scope('move_to'):
        with tf.name_scope('input'):
            board_t = tf.placeholder(tf.float32, [None, 36])
            owner_t = tf.placeholder(tf.float32, [None, 36])
            move_from_t = tf.placeholder(tf.float32, [None])
            move_to_one_hot_t = tf.placeholder(tf.float32, [None, 36])

            shaped_board = tf.reshape(board_t, [-1,6,6])
            shaped_owner = tf.reshape(owner_t, [-1,6,6])
            shaped_state = tf.stack([shaped_board, shaped_owner], axis=-1)

        with tf.name_scope('layer_1'):
            W_conv1 = weight_variable([4, 4, 2, 32]) # 4x4, 2 channel input, 32 channel output of 6x6es
            b_conv1 = bias_variable([32])

            h_conv1 = tf.nn.relu(conv2d(shaped_state, W_conv1) + b_conv1)

        with tf.name_scope('layer_2'):
            W_conv2 = weight_variable([4, 4, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        with tf.name_scope('connected_layer'):
            W_fc1 = weight_variable([6*6*64, 1024])
            b_fc1 = bias_variable([1024])
            h_conv2_flat = tf.reshape(h_conv2, [-1, 6*6*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('readout'):
            W_fc2 = weight_variable([1024, 36])
            b_fc2 = bias_variable([36])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=move_to_one_hot_t))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(move_to_one_hot_t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        samples = len(board)
        loc = 0
        for i in range(iterations):
            batch_board = board[loc:loc+BATCH_SIZE]
            batch_owner = owner[loc:loc+BATCH_SIZE]
            batch_move_from = move_from[loc:loc+BATCH_SIZE]
            batch_move_to_one_hot = move_to_one_hot[loc:loc+BATCH_SIZE]

            # print(batch_board)
            # exit()

            loc = (loc + BATCH_SIZE) % samples
            if (i-1) % 1000 == 0 or i < 10:
                train_accuracy = accuracy.eval(feed_dict={
                    board_t: batch_board, owner_t: batch_owner, move_from_t: batch_move_from, move_to_one_hot_t: batch_move_to_one_hot, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))

            train_step.run(feed_dict={board_t: batch_board, owner_t: batch_owner, move_from_t: batch_move_from, move_to_one_hot_t: batch_move_to_one_hot, keep_prob: 0.5})


        model_name = 'move_to'
        dir_save = dir_location(model_name)
        saver.save(sess, os.path.join(dir_save, model_name))
        print('Saved move_to model to:', dir_save)



def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def import_data():
    return pandas.read_pickle(CSV_LOCATION)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('iterations', default='10000')
    args = parser.parse_args()

    data = import_data()
    # data columns are board, visible, owner, movement, move_from, move_to, board_size

    if args.model == 'from':
        train_move_from(data['board'].tolist(), data['owner'].tolist(), data['move_from_one_hot'].tolist(), int(args.iterations))
    if args.model == 'to':
        train_move_to(data['board'].tolist(), data['owner'].tolist(), data['move_from'].tolist(), data['move_to_one_hot'].tolist(), int(args.iterations))