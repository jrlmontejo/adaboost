#!/usr/bin/env python3

"""Adaboost implementation

Compiler: Python 3.6.5
OS: macOS 10.14
"""

import numpy as np
import perceptron

def adabtrain(training_set, K = 1000):
  """Trains the Adaboost classifier

  Parameters:
    training_set: The training set
    K: The number of iterations/learners

  Returns:
    The generated alpha values and weight vectors
  """

  # number of training examples
  N = len(training_set)

  # initial weights
  w = np.full(N, 1 / N)

  # initialize alpha and weight vectors
  alphas = []
  params = []

  for t in range(0, K):
    # get sample w/ replacement S according to w
    S = np.take(training_set, np.random.choice(N, N, p = w), axis = 0)

    # train S using perceptron
    weights = perceptron.classify(S)
    params.append(weights)

    # classify training set
    expected, predicted = perceptron.predict(training_set, weights)

    # compute deltas
    deltas = (expected != predicted).astype(int)

    # compute error
    epsilon = np.dot(w, deltas)

    # compute coefficient
    alpha = 0.5 * np.log((1 - epsilon) / epsilon)
    alphas.append(alpha)

    # compute new weights
    w_hat = w * np.exp(-alpha * expected * predicted)
    Z = np.sum(w_hat)
    w = w_hat / Z

    print('Error @ iteration {}: {}'.format(t + 1, epsilon))

  return alphas, params

def adabpredict(test_set, alphas, params):
  """Predicts the labels using the Adaboost classifier

  Parameters:
    test_set: The test set
    alphas: The generated alpha values
    params: The model parameters i.e. weight vectors

  Returns:
    The list of accuracies over K learners where K = 10, 20, ..., 1000
  """

  N = len(test_set)

  # initialize number of learners to track
  K = np.arange(10, 1010, step = 10)

  # initialize storage
  accuracies = []
  expected = []
  predicted = []

  print('Running predictions on test set...')

  for instance in test_set:
    x, y = perceptron.get_input(instance)
    h = np.array([perceptron.get_label(x, w) for w in params])
    sums = np.cumsum(alphas * h)
    y_hats = np.sign(sums)

    expected.append(y)
    predicted.append(y_hats)

  for k in K:
    y = np.array(expected)
    h = np.array([p[k - 1] for p in predicted])
    acc = np.count_nonzero(y == h) / N
    accuracies.append(acc)

    print('Accuracy @ K = {}: {}'.format(k, acc))

  return accuracies

def adabrun(dataset_file, train_size, test_size):
  """Runs the Adaboost training and classification on the given dataset

  Parameters:
    dataset_file: The dataset filename
    train_size: The size of training set
    test_size: The size of test set
  """

  print('** START ADABOOST **\n')

  print('Generating training and test sets from {}...'.format(dataset_file))
  dataset = np.genfromtxt(dataset_file, delimiter = ',')
  np.random.shuffle(dataset)

  training_set = dataset[:train_size]
  test_set = dataset[train_size:train_size + test_size]

  print('\nTraining on {}...'.format(dataset_file))
  alphas, params = adabtrain(training_set)

  print('\nTesting classifier on {} training set...'.format(dataset_file))
  train_accuracies = adabpredict(training_set, alphas, params)

  print('\nTesting classifier on {} test set...'.format(dataset_file))
  test_accuracies = adabpredict(test_set, alphas, params)

  train_output = 'train_accuracies_{}'.format(dataset_file)
  np.savetxt(train_output, train_accuracies, delimiter = ',', fmt = '%f')
  print('Accuracies saved to {}'.format(train_output))

  test_output = 'test_accuracies_{}'.format(dataset_file)
  np.savetxt(test_output, test_accuracies, delimiter = ',', fmt = '%f')
  print('Accuracies saved to {}'.format(test_output))

  print('\n** END ADABOOST **\n')

def start():
  """Main
  """

  adabrun('banana_data.csv', 400, 4900)
  adabrun('splice_data.csv', 1000, 2175)

# run when not imported
if __name__ == "__main__":
  start()