#!/usr/bin/env python3
import tensorflow as tf
import numpy as np 
from tensorflow.python.ops import rnn, rnn_cell

import unstructured_data as data 
from util import one_hot_array

words, domain = data.load()


# Parameters
learning_rate = 0.0001
training_iters = 1000000
batch_size = 20
display_step = 10


# Network Parameters
n_input = len(domain)   # language domain
n_steps = 50            # sequence length
n_hidden = 50           # hidden layer num of features
n_classes = len(domain) # output domain


def next_batch():
  start =  np.random.randint(len(domain)-batch_size*n_steps-1)
  batch_x = []
  batch_y = []

  for i in range(batch_size):
    seq = start + n_steps * i 
    batch_x += [one_hot_array(domain, words[seq:seq+n_steps])]
    batch_y += [one_hot_array(domain, words[seq+1:seq+n_steps+1])]

  print('x: ({},{},{})'.format(len(batch_x), len(batch_x[0]), len(batch_x[0][0])))
  return batch_x, batch_y


# Input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
W = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

def LSTM():
  xT = tf.transpose(x, [1,0,2])
  xR = tf.reshape(xT, [-1, n_input])

  # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
  xS = tf.split(0, n_steps, xR)

  # Define a lstm cell with tensorflow
  lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

  # Get lstm cell output
  outputs, states = rnn.rnn(lstm_cell, xS, dtype=tf.float32)

  # Linear activation, using rnn inner loop last output
  return tf.matmul(outputs[-1], W) + b


pred = LSTM()

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Launch the graph
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    step = 1

    while step * batch_size < training_iters:
        
        batch_x, batch_y = next_batch()
        sess.run(optimizer, feed_dict={x: batch_x, y: [0]})

        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_size = 100
    batch_x, batch_y = next_batch()
    result = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
    print("Testing Accuracy:", result)
