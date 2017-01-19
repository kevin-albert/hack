#!/usr/bin/env python3
from os import listdir
from os.path import isfile, join
import re
import numpy as np

from lxml import etree



def load(dir='./TNG_DATA'):
  """
  Get an array of words, like:
  [ 'PICARD:', 'You', 'will', 'agree', ',', 'Data', ',', 'that', "Starfleet's", 'orders', 'are', 'difficult', '?',
    'DATA:', 'Difficult', '?', 'Simply', 'solve', 'the', 'mystery', 'of', 'Farpoint', 'Station', '.',
    ...
  ]
  """

  words = []
  total_lines = 0

  #data_files = [f for f in listdir(dir) if re.match(r'[0-9]+.htm', f)]
  data_files = ['101.htm']

  print('Parsing {} HTML files'.format(len(data_files)))
  for data_file in data_files:
    html = etree.parse(open(join(dir, data_file)), etree.HTMLParser())
    for text in html.xpath('//table/tbody/tr/td//text()'):
      text = text.strip()
      if re.match(r'^[A-Z]+( .*)?:', text):
        # spoken line
        words += parse_line(text)
        total_lines += 1

  domain = list(set(words))
  domain.sort()
  print('Lines:         {}'.format(total_lines))
  print('Words:         {}'.format(len(words)))
  print('Unique words:  {}'.format(len(domain)))
  return words, domain


def parse_line(line):
  actor = re.sub(' .*', '', line.split(':')[0]).strip().upper()+':'
  text_and_parens = line.split(':', 1)[1].strip().lower()
  text = re.sub(r'\([^\)]*\)', '', text_and_parens)
  words = re.findall(r'([a-zA-Z0-9\']+|[\.\?!,])', text)
  return [actor] + words + ['(STOP)']

