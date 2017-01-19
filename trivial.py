#!/usr/bin/env python3
import tensorflow as tf
import numpy as np 
from tensorflow.python.ops import rnn, rnn_cell
from structured_data import load
from util import one_hot_array

data = load()

n_input = len(domain)
n_output = len(actors)
n_steps = 3
batch_size = 2
n_hidden = 100
learning_rate = 0.01


x = tf.placeholder(tf.float32, [None, n_steps, n_input])
xT = tf.transpose(x, [1,0,2])
xR = tf.reshape(x, [-1, n_input])
xS = tf.split(0, n_steps, xR)

y_ = tf.placeholder(tf.float32, [None, n_output])

lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, states = rnn.rnn(lstm_cell, xS, dtype=tf.float32)

# weights, biases
W = tf.Variable(tf.random_normal([n_hidden, n_output]))
b = tf.Variable(tf.random_normal([n_output]))

# get output
y = tf.matmul(outputs[-1], W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  batch_x = [[[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0]],
             [[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0]]]
  batch_y = [[0,1],[0,1]]

  for _ in range(10):
    session.run(optimizer, feed_dict={x: batch_x, y_: batch_y})
    loss = session.run(cost, feed_dict={x: batch_x, y_: batch_y})
    print('Loss: {}'.format(loss))
