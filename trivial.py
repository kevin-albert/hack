#!/usr/bin/env python3
import tensorflow as tf
import numpy as np 
from tensorflow.python.ops import rnn, rnn_cell
from structured_data import load_data
from util import one_hot_array

tng_data = load_data()

n_input = tng_data.get_num_words()
n_output = tng_data.get_num_people()
n_steps = tng_data.get_seq_length()
batch_size = 50
n_hidden = 100
learning_rate = 0.1
training_iters = 1000000

print('Initializing Tensorflow')

# LSTM -> Feedforward layer 
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
xT = tf.transpose(x, [1,0,2])
xR = tf.reshape(x, [-1, n_input])
xS = tf.split(0, n_steps, xR)


lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, states = rnn.rnn(lstm_cell, xS, dtype=tf.float32)
# gru_cell = rnn_cell.GRUCell(n_hidden)
# outputs, states = rnn.rnn(gru_cell, xS, dtype=tf.float32)

# weights, biases
W = tf.Variable(tf.random_normal([n_hidden, n_output]))
b = tf.Variable(tf.random_normal([n_output]))

# get output
y = tf.matmul(outputs[-1], W) + b

# placeholder for expected output
y_ = tf.placeholder(tf.float32, [None, n_output])

# optimization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# reporting
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('Training')
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  step = 1

  while step * batch_size < training_iters:
    batch_x, batch_y = tng_data.next_batch(batch_size)
    feed = {x: batch_x, y_: batch_y}

    session.run(optimizer, feed_dict=feed)

    # print stats along the way
    if step % 100 == 0:
      # get batch mean accuracy
      acc = session.run(accuracy, feed_dict=feed)
      
      # get batch cost
      loss = session.run(cost, feed_dict=feed)
      print('Step: {:10}, Loss: {:0.8f}, Accuracy: {:0.8f}'\
            .format(step * batch_size, loss, 0))
    step += 1

print('Done')
