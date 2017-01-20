#!/usr/bin/env python3
from os import listdir
from os.path import isfile, join
import re
import numpy as np

from lxml import etree

GO    = '<GO>'
TITLE = '<TITLE>'
SCENE = '<SCENE>'
EOL   = '<EOL>'
STOP  = '<STOP>'
PAD   = '<PAD>'

def load(dir='./TNG_DATA'):
  """
  Get an array of words, like:
  [ 
    '(GO)', '(EPISODE)', 'encounter, 'at, 'farpoint', '(SCENE)', 
    'PICARD:', 'You', 'will', 'agree', ',', 'Data', ',', 'that', "Starfleet's", 
    'orders', 'are', 'difficult', '?', 'DATA:', 'Difficult', '?', 'Simply', 'solve', 'the', 'mystery', 'of', 'Farpoint', 'Station', '.',
    ...
  ]
  """

  data = []
  domain = set()
  data_files = [f for f in listdir(dir) if re.match(r'\.html$', f)]
  #data_files = ['NextGen_101.html']

  print('Parsing {} HTML files'.format(len(data_files)))
  for data_file in data_files:
    print('Parsing ' + data_file)
    html = etree.parse(open(join(dir, data_file)), etree.HTMLParser())
    words = [GO]

    # parse out title element
    title = re.sub(r'\s+', ' ', html.xpath('/html/body/p/font/b/text()')[0])
    
    words += [TITLE] + re.findall(r'([a-zA-Z0-9\']+|[\.\?!,])', title)

    for text in html.xpath('//table/tbody/tr/td//text()'):
      # skip empty text
      text = text.strip()
      if text:
        if re.match(r'^\s*\[[^\]]*\].*', text):
          words += [SCENE]
        words += re.findall(r'([a-zA-Z0-9\']+|[\.\?!,\(\)])', text)
        words += [EOL]
    words += [STOP]
    domain += set(words)
    data += [words]

  # domain = list(set(words))
  # domain.sort()
  # print('Lines:         {}'.format(total_lines))
  # print('Words:         {}'.format(len(words)))
  # print('Unique words:  {}'.format(len(domain)))
  # return words, domain
  return data


class StarTrekData:
  def __init__(self, data, domain):
    self.data = data