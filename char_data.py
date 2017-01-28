#!/usr/bin/env python3
from os import listdir
from os.path import isfile, join
import re
import numpy as np
from lxml import etree

from util import one_hot, one_hot_array

GO    = '^'
SEP   = '|'
EOL   = '\n'
STOP  = '*'
PAD   = '_'


domain = 0
encoding_array = [19 for _ in range(255)]
decoding_array = ['?' for _ in range(255)]

for o in range(255):
  c = chr(o)
  if re.match(r'[a-zA-Z0-9 \n\[\]\(\)\.,\?!é\^:\*_-]', c):
    encoding_array[o] = domain
    decoding_array[domain] = c
    domain += 1


data = []

def _load(dir='./_DATA'):

  #data_files = [f for f in listdir(dir) if re.match(r'.*\.html$', f)]
  # Uncomment for simple version (just one TNG episode)
  data_files = ['NextGen_186.html']

  print('Parsing {} HTML files'.format(len(data_files)))
  for data_file in data_files:
    html = etree.parse(open(join(dir, data_file)), etree.HTMLParser())
    words = GO

    # parse out title element
    words += _parse_line(html.xpath('//body//b/text()')[0])

    for text in html.xpath('//table/tbody/tr/td//text()'):
      # skip empty text
      text = text.strip()
      if text:
        words += _parse_line(text)
        if len(words) >= 200:
          break

    words += STOP
    data.append(words)
  print('Text length: {}'.format(len(words)))

def start_token():
  return encode(GO)


def start_seq(seq_length):
  return encode_string(data[0][0:seq_length])


def _parse_line(line):
  line = re.sub(r'\s+', ' ', line)
  return re.sub(r'[^a-zA-Z0-9 \n\[\]().,?!é\:-]', '', line) + EOL


def encode(char):
  x = encoding_array[ord(char)]
  return one_hot(domain, x)


def decode(array):
  x = np.argmax(array)
  return decoding_array[x]


def encode_string(str):
  return list(map(encode,str))


def decode_string(arrays):
  return ''.join(map(decode, arrays))

def iterate(batch_size, seq_length, shuffled=False):

  if len(data) == 0:
    _load()

  order = range(len(data))
  if shuffled:
    order = [order]
    np.random.shuffle(order)

  batch_x = []
  batch_y = []
  for title_idx in order:
    title = data[title_idx]
    seq_start = 0
    while seq_start < len(title):
      sequence = title[seq_start:seq_start+seq_length+1]
      sequence += PAD * (seq_length + 1 - len(sequence))
      seq_start += seq_length
      batch_x += [encode_string(sequence[:-1])]
      batch_y += [encode_string(sequence[1:])]
      if len(batch_x) >= batch_size:
        yield batch_x, batch_y
        batch_x = []
        batch_y = []

