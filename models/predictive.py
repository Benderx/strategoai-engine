import tensorflow as tf
import pandas

CSV_LOCATION = 'games'
BATCH_SIZE = 50

# for now, just use full board state and predict piece chosen
def train_model(data, labels):
    sess = tf.InteractiveSession()
    with tf.name_scope('input'):
        full_state = tf.placeholder(tf.float32, [None, 36])
        move_taken = tf.placeholder(tf.float32, [None, 36])
        shaped_state = tf.reshape(full_state, [-1, 6, 6, 1])

    with tf.name_scope('layer_1'):
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(shaped_state, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('layer_2'):
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('connected_layer'):
        W_fc1 = weight_variable([256, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('readout'):
        W_fc2 = weight_variable([1024, 36])
        b_fc2 = bias_variable([36])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=move_taken))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(move_taken, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    samples = len(board)
    print(len(board))
    for i in range(5000):
        loc = 0
        batch = data[loc:loc+BATCH_SIZE]
        batch_labels = labels[loc:loc+BATCH_SIZE]
        loc = (loc + 50) % samples
        if (i-1) % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                full_state: batch, move_taken: batch_labels, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        train_step.run(feed_dict={full_state: batch, move_taken: batch_labels, keep_prob: 0.5})




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
    data = import_data()
    board = [item for sublist in data[['board']].as_matrix() for item in sublist]
    moves = data[['move_from']]
    labels = []
    for move in moves.as_matrix():
        temp = [0]*36
        temp[move[0]] = 1
        labels.append(temp)

    train_model(board, labels)