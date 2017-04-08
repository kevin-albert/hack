#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell
import char_data as data


n_input = data.domain
n_output = data.domain
n_steps = 20
batch_size = 1
n_hidden = 200
n_layers = 3
learning_rate = 0.001
epochs = 50

print('Initializing Tensorflow')


##########################################################################
# LSTM -> Feedforward layer                                              #
##########################################################################
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
xT = tf.transpose(x, [1, 0, 2])
xR = tf.reshape(xT, [-1, n_input])
xS = tf.split(xR, n_steps)

lstm = BasicLSTMCell(n_hidden, forget_bias=1.0)
cell = MultiRNNCell([lstm] * n_layers)
outputs, states = rnn.static_rnn(cell, xS, dtype=tf.float32)

# weights, biases
W = tf.Variable(tf.random_normal(
    [n_hidden, n_output], stddev=1.0 / np.sqrt(n_hidden)))
b = tf.Variable(tf.random_normal([n_output]))

# ok, here we get an array of length n_steps with tensors of shape
# (batch, n_outputs).
Y = []
for output in outputs:
    Y += [tf.matmul(output, W) + b]


##########################################################################
# placeholder for expected output                                        #
##########################################################################
Y_ = tf.placeholder(tf.float32, [None, n_steps, n_output])
# transpose and unstack to get (n_steps, batch_size, n_output) where the first
# dimension is an array (not a tensor).
Y_U = tf.unstack(Y_, axis=1)


##########################################################################
# cross-entropy for each step                                            #
##########################################################################
ce = []
for y, y_ in zip(Y, Y_U):
    ce += [tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_), [-1])]

# compute cost, optimize
cost = tf.reduce_mean(tf.stack(ce, 1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


print('Training')
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    batch = 1
    for epoch in range(epochs):
        for batch_x, batch_y in data.iterate(batch_size, n_steps):

            feed = {x: batch_x, Y_: batch_y}
            # print stats along the way
            if batch % 500 == 0:
                # get batch cost
                loss = session.run(cost, feed_dict=feed)
                print('Epoch: {:5}, Batch: {:8}, Step: {:11}, Loss: {:0.5f}'
                      .format(epoch, batch, batch * batch_size, loss))

                line = ''
                for output in Y:
                    char = session.run(output, feed_dict=feed)
                    line += data.decode(char[0])
                print('expected: {}'.format(data.decode_string(batch_y[0])))
                print('actual:   {}'.format(line))

            session.run(optimizer, feed_dict=feed)

            batch += 1

    print('Done')
    exit()

    # Test it out!
    print('Sampling')
    print('*' * 80)

    limit = 100
    seq = data.start_seq(n_steps)
    Yp = tf.squeeze(tf.reshape(tf.pack(Y), [1, n_steps, n_output]), [0])
    sample_step = 0

    while True:
        seq = session.run(Yp, feed_dict={x: [seq]})
        print(data.decode_string(seq), end='')
        if sample_step >= limit:
            break
        sample_step += n_steps

    # while True:
    #   t_now = sample_step % n_steps
    #   t_next = (sample_step+1) % n_steps
    #   result = session.run(Y[t_now], feed_dict={x: [seq]})
    #   char = data.decode(result[0])
    #   print(char, end='')
    #   if char == data.STOP or sample_step >= limit:
    #     break
    #   seq[t_next] = result[0]
    #   sample_step += 1

    print('')
    print('*' * 80)


print('Done')
