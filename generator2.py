#!/usr/bin/env python3
import tensorflow as tf
import numpy as np 
from tensorflow.python.ops import rnn, rnn_cell
import char_data as data


def RNN:

  def __init__(self, batch_size, n_steps):
    lstm = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    cell = rnn_cell.MultiRNNCell([lstm] * n_layers)
    self.initial_state = cell.zero_state(batch_size, tf.float32)
    outputs = []
    with tf.variable_scope("RNN"):
      for step in range(n_steps):
        if step > 0: 
          tf.get_variable_scope().reuse_variables()
        


  def rnn_seq(rnn, batch_size, n_steps):
    outputs = []
    state = tf.zeros([batch_size, self.lstm_state_size])
    for _ in n_steps:
      output, state = lstm()


  def rnn_run(rnn, session, x):
    state = tf.zeros([1, lstm.state_size])
    while True:
      x, state = self.lstm(x, state)
      yield session.run(x)



n_input = data.domain
n_output = data.domain
n_steps = 20
batch_size = 1
n_hidden = 100
n_layers = 2
learning_rate = 0.001
epochs = 1000

print('Initializing Tensorflow')


rnn = RNN(lstm_stack, batch_size, n_steps)

################################################################################
# LSTM -> Feedforward layer                                                    #
################################################################################
x = tf.placeholder(tf.float32, [None, n_input])



rnn_in = x
for step in range(n_steps):
  
