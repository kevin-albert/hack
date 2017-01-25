#!/usr/bin/env python3
from os import listdir
from os.path import isfile, join
import re
import numpy as np
from lxml import etree

from util import one_hot, one_hot_array


GO    = '<GO>'
TITLE = '<TITLE>'
EOL   = '<EOL>'
STOP  = '<STOP>'
PAD   = '<PAD>'

def load_star_trek_data(dir='./_DATA'):
  """
  Get an array of words, like:
  [ 
    '<GO>', '<TITLE>', 'encounter, 'at, 'farpoint', '[', 'bridge', 
    ']', 'picard', ':', 'you', 'will', 'agree', ',', 'data', ',', 'that', 
    "starfleet's", 'orders', 'are', 'difficult', '?', '<EOL>', 'data', ':', 
    'difficult', '?', 'simply', 'solve', 'the', 'mystery', 'of', 'farpoint', 
    'station', '.', '<EOL>', ... '<STOP>'
    ...
  ]
  """

  data = []
  domain = {GO, TITLE, EOL, STOP, PAD}
  #data_files = [f for f in listdir(dir) if re.match(r'.*\.html$', f)]
  # Uncomment for simple version (just one TNG episode)
  data_files = ['NextGen_101.html']

  print('Parsing {} HTML files'.format(len(data_files)))
  for data_file in data_files:
    html = etree.parse(open(join(dir, data_file)), etree.HTMLParser())
    words = [GO]

    # parse out title element
    title = re.sub(r'\s+', ' ', html.xpath('//body//b/text()')[0])
    # print('File: {}, Title: {}'.format(data_file, title))

    words += [TITLE] + re.findall(r'([a-zA-Z0-9\']+|[\.\?!,])', title)

    for text in html.xpath('//table/tbody/tr/td//text()'):
      # skip empty text
      text = text.strip()
      if text:
        # if re.match(r'^\s*\[[^\]]*\].*', text):
        #   words += [SCENE]
        words += re.findall(r'([a-zA-Z0-9\']+|[\.\?!,\(\):])', text)
        words += [EOL]
        if len(words) >= 50:
          break

    words += [STOP]
    domain.update(words)
    data += [words]

  print('Unique words:  {}'.format(len(domain)))
  return StarTrekData(data, domain)


class StarTrekData:

  def __init__(self, data, domain):
    self.encoder = {}
    self.decoder = list(sorted(list(domain)))
    for i, token in enumerate(self.decoder):
      self.encoder[token] = i

    self.titles = []
    for title in data:
      encoded = [self.encoder[token] for token in title]
      self.titles += [encoded]


  def start_token(self):
    """ feed this in first """
    return self.encode(GO)


  def encode(self, token):
    """ Convert a token from a string to a one-hot array """
    return one_hot(len(self.encoder), self.encoder[token])


  def decode(self, encoded_word):
    """ Convert a one-hot array to a token """ 
    return self.decoder[np.argmax(encoded_word)]


  def decode_array(self, encoded_array):
    """ Apply self.decode to an array """
    return [self.decode(word) for word in encoded_array]


  def domain(self):
    """ How many words do I know? """
    return len(self.decoder)


  def batch_iterate(self, n_batches, batch_size, seq_length, shuffled=False):
    return BatchIterator(self, n_batches, batch_size, seq_length, shuffled)


class BatchIterator:

  def __init__(self, data, epochs, batch_size, seq_length, shuffled):
    self.data = data
    self.shuffled = shuffled
    self.epochs = epochs
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.epoch = 0
    self.title = 0
    self.seq = 0
    self.order = [i for i in range(len(data.titles))]
    self.done = False
    if self.shuffled:
      np.random.shuffle(self.order)


  def __iter__(self):
    self.epoch = 0
    self.title = 0
    self.seq = 0
    return self 


  def __next__(self):

    if self.done:
      raise StopIteration

    """
    returns batch_x, batch_y such that:
    batch_x is a list of sequences of tokens, and
    batch_y is the same as batch_x except that each sequence starts one word 
    later.
    tokens are represented as one-hot arrays
    """

    batch_x = []
    batch_y = []

    for _ in range(self.batch_size):
      index = self.order[self.title]
      title = self.data.titles[self.order[self.title]]
      sequence = title[self.seq:self.seq + self.seq_length + 1]
      if len(sequence) <= self.seq_length:
        # This means we've reached the end of a title
        # pad the current sequence (it should already have a <STOP>) and then 
        # move on to the next title
        sequence += [self.data.encoder[STOP]] * (self.seq_length + 1 - len(sequence))
        self.title += 1
        self.seq = 0
        if self.title >= len(self.order):
          self.epoch += 1
          if self.epoch >= self.epochs:
            # We've gone through all 
            self.done = True
            break
          else:
            # This means we've gone through all titles. Loop back to the first 
            # title and shuffle if necessary
            self.title = 0
            self.epoch += 1
            if self.shuffled:
              np.random.shuffle(self.order)
      else:
        self.seq += 1

      # add the sequence to the batch, with y starting one token after x
      batch_x += [one_hot_array(self.data.domain(), sequence[:-1])]
      batch_y += [one_hot_array(self.data.domain(), sequence[1:])]
    
    if self.done and len(batch_x) == 0:
      # we ran out of data
      raise StopIteration

    return batch_x, batch_y


