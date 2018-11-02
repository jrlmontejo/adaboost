#!/usr/bin/env python3

"""Accuracy Plot over K Learners

Compiler: Python 3.6.5
OS: macOS 10.14
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_accuracies(train_file, test_file, fig):
  """Plots the accuracies of the adaboost classifier over K learners, K = 10, 20, ..., 1000

  Parameters:
    train_file: The file containing the training set prediction accuracies
    test_file: The file containing the test set prediction accuracies
  """

  train_acc = np.genfromtxt(train_file)
  test_acc = np.genfromtxt(test_file)

  points = np.arange(len(train_acc))
  K = points * 10 + 10

  plt.figure(fig)
  plt.plot(K, train_acc, label = 'Training', linewidth = 2)
  plt.plot(K, test_acc, label = 'Test', linewidth = 2)
  plt.xlabel('Number of Learners (K)')
  plt.ylabel('Accuracy')
  plt.legend()

def start():
  """Main
  """

  banana_train = 'train_accuracies_banana_data.csv'
  banana_test = 'test_accuracies_banana_data.csv'

  plot_accuracies(banana_train, banana_test, 1)

  splice_train = 'train_accuracies_splice_data.csv'
  splice_test = 'test_accuracies_splice_data.csv'

  plot_accuracies(splice_train, splice_test, 2)

  plt.show()

# run when not imported
if __name__ == "__main__":
  start()