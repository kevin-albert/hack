#!/usr/bin/env python3
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import util

from lxml import etree

STOP = '[STOP]'
PAD = '[PAD]'

def load_data(dir='./TNG_DATA', seq_length=40):
  """
  when choosing seq_length, keep in mind the distribution of sequence sizes in 
  the raw data:
  0-9:      40%
  10-19:    33%
  20-29:     8%
  30-39:     5%
  40-99:     5%
  100-499:  <1%
  min is 1
  max is 477

  default seq_length is 40

  returns 3 variables:
  people = {'PICARD', 'DATA', ...},
  domain = {'you','station','simply',...},
  lines = [
    ('PICARD', ['You', 'will', 'agree', ',', 'Data', ',', 'that', "Starfleet's", 'orders', 'are', 'difficult', '?', '(STOP)']),
    ('DATA', ['Difficult', '?', 'Simply', 'solve', 'the', 'mystery', 'of', 'Farpoint', 'Station', '.', '(STOP)']),
    ...
  ]
  """
  lines = []
  people = set()
  domain = [PAD,STOP]
  scenes = []

  # data_files = [f for f in listdir(dir) if re.match(r'[0-9]+.htm', f)]
  data_files = ['101.htm']
  print('Parsing {} HTML files'.format(len(data_files)))
  for data_file in data_files:
    html = etree.parse(open(join(dir, data_file)), etree.HTMLParser())
    for text in html.xpath('//table/tbody/tr/td//text()'):
      text = text.strip()
      if re.match(r'^[A-Z]+( .*)?:', text):
        # spoken line
        full_line = parse_line(text)
        domain += full_line[1]
        people.add(full_line[0])
        # split into evenly-size sequences
        for i in range(0, len(full_line), seq_length):
          words = full_line[1][i:i+seq_length] 
          words += [PAD] * (seq_length - len(words))
          lines.append((full_line[0], words))

      elif re.match(r'^\[.*\]', text):
        # track scenes so we can choose intelligent batches
        scenes += [(text, len(lines))]

  domain = sorted(list(set(domain)))
  people = sorted(list(people))
  print('Lines:         {}'.format(len(lines)))
  print('Unique words:  {}'.format(len(domain)))
  print('Actors:        {}'.format(len(people)))
  return TNGData(seq_length, people, domain, lines, scenes) 


def parse_line(line):
  person = re.sub(' .*', '', line.split(':')[0]).strip().upper()
  text_and_parens = line.split(':', 1)[1].strip().lower()
  text = re.sub(r'\([^\)]*\)', '', text_and_parens)
  words = re.findall(r'([a-zA-Z0-9\']+|[\.\?!,])', text) + ['(STOP)']
  return person, words


class TNGData:
  def __init__(self, seq_length, people, words, lines, scenes):
    self.seq_length = seq_length
    self.people = people
    self.words = words
    self.lines = []
    self.scenes = scenes
    print('Encoding...')

    words_map = {}
    for i, word in enumerate(words):
      words_map[word] = i 

    for line in lines:
      person_enc = util.binary_search(people, line[0])
      words_enc = [ words_map[word] for word in line[1] ]
      self.lines.append((person_enc, words_enc))

    print('Done encoding')

  def get_seq_length(self):
    return self.seq_length


  def get_num_words(self):
    return len(self.words)


  def get_num_people(self):
    return len(self.people)


  def next_batch(self, batch_size, start_idx=None):
    """Get batch_x, batch_y such that:
    shape of batch_x is (batch_size, seq_length, |words|)
    shape of batch_y is (batch_size, |people|)

    The batch is a contiguous list of sequences. If start_idx is supplied, then 
    we start there. Otherwise we choose a scene randomly (weighted by length)
    and start at the beginning.
    """
    if start_idx == None:
      start = np.random.randint(0, len(self.lines)-batch_size)
      start_scene = self.find_scene_near(start)
      start_idx = start_scene[1]

    batch_x = []  # array of one-hot encoded lines
    batch_y = []  # array of one-hot encoded 
    for line in self.lines[start_idx:start_idx+batch_size]:
      batch_y += [util.one_hot(len(self.people), line[0])]
      batch_x += [util.one_hot_array(len(self.words), line[1])]

    return batch_x, batch_y


  def find_scene_near(self, line):
    scenes = self.scenes
    if line > scenes[-1][1]:
      return scenes[-1]
    elif line < scenes[0][1]:
      return scenes[0]

    lo = 0
    hi = len(scenes)
    while True:
      mid = (lo + hi) // 2
      if scenes[mid][1] <= line:
        if mid == len(scenes)-1 or scenes[mid+1][1] > line:
          return scenes[mid]
        else:
          lo = mid 
      elif scenes[mid][1] > line:
        if mid == 0:
          return scenes[0]
        else:
          hi = mid 


  def decode_word(self, word):
    return self.words[np.argmax(word)]


  def decode_sequence(self, sequence):
    result = [self.decode_word(word) for word in sequence]
    return list(filter(lambda word: word != PAD, result))


  def decode_person(self, person):
    return self.people[np.argmax(person)]
