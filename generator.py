#!/usr/bin/env python3
import tensorflow as tf
import numpy as np 
from tensorflow.python.ops import rnn, rnn_cell
import char_data as data


################################################################################
# load the data                                                                #
################################################################################

n_input = data.domain
n_output = data.domain
n_steps = 50
batch_size = 1
n_hidden = 100
n_layers = 2
learning_rate = 0.005
epochs = 200

print('Initializing Tensorflow')


################################################################################
# LSTM -> Feedforward layer                                                    #
################################################################################
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
xT = tf.transpose(x, [1,0,2])
xR = tf.reshape(x, [-1, n_input])
xS = tf.split(0, n_steps, xR)

lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
lstm_stack = rnn_cell.MultiRNNCell([lstm_cell] * n_layers)
outputs, states = rnn.rnn(lstm_stack, xS, dtype=tf.float32)

# weights, biases
W = tf.Variable(tf.random_normal([n_hidden, n_output], stddev=1.0/np.sqrt(n_hidden)))
b = tf.Variable(tf.random_normal([n_output]))

# ok, here we get an array of length n_steps with tensors of shape 
# (batch, n_outputs).
Y = []
for output in outputs:
  Y += [tf.matmul(output, W) + b]


################################################################################
# placeholder for expected output                                              #
################################################################################
Y_ = tf.placeholder(tf.float32, [None, n_steps, n_output])
# transpose and unstack to get (n_steps, batch_size, n_output) where the first
# dimension is an array. 
Y_U = tf.unstack(Y_, axis=1)


################################################################################
# cross-entropy for each step                                                  #
################################################################################
ce = []
for y, y_ in zip(Y, Y_U):
  ce += [tf.reshape(tf.nn.softmax_cross_entropy_with_logits(y, y_), [-1])]

# compute cost, optimize
cost = tf.reduce_mean(tf.stack(ce,1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
out = tf.reshape(tf.slice(Y[0],[0,0],[1,n_output]),[-1])

print('Training')
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  batch = 1
  for epoch in range(epochs):
    for batch_x, batch_y in data.iterate(batch_size, n_steps):

      feed = {x: batch_x, Y_: batch_y}
      # print stats along the way
      if batch % 100 == 0:
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

  # Test it out!
  print('Testing')
  print('*' * 80)
  seq = [data.start_token()] + [np.zeros(n_input)] * (n_steps-1)
  limit = 100
  while True:
    result = session.run(Y[0], feed_dict={x: [seq]})
    char = data.decode(result[0])
    if char == data.STOP or limit == 0:
      break
    print(char, end='')
    seq[0] = result[0]
    limit -= 1

  print('*' * 80)
  

print('Done')


