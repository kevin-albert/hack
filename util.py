import numpy as np


def binary_search(domain, value):
  lo = 0
  hi = len(domain)
  while True:
    mid = (lo + hi) // 2
    if value < domain[mid]:
      hi = mid
    elif value > domain[mid]:
      lo = mid
    else:
      return mid
    if hi == lo:
      return None


def one_hot(domain, value):
  A = np.zeros(domain)
  A[value] = 1
  return A 


def one_hot_array(domain, values):
  return [one_hot(domain, value) for value in values]