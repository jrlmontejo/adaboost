#!/usr/bin/env python3

"""Perceptron classifier using Pocket Algorithm

Compiler: Python 3.6.5
OS: macOS 10.14
"""

import numpy as np

def add_labels(data, label):
  """Adds labels to data

  Parameters:
    data: The array of data with label label
    label: The label to append

  Returns:
    The dataset with labels
  """

  size = len(data)
  labels = np.full(size, label).reshape(size, 1)
  return np.hstack((labels, data))

def divide_dataset(data):
  """Divides the dataset into two subsets

  Parameters:
    data: The dataset to divide

  Returns:
    The two subsets
  """

  size = len(data)
  divider = int(size / 2)
  m, n = data[:divider], data[divider:]
  return m, n

def generate_data(mu, size):
  """Generates data from normal distribution with mean = mu and standard deviation = 1

  Parameters:
    mu: The mean of the distribution
    size: The number of data points to generate

  Returns:
    The generated data points
  """

  samples = np.random.normal(mu, 1, (size, len(mu)))
  return samples

def create_dataset():
  """Creates the training and test sets

  Returns:
    The training set and test sets
  """

  N = 100

  p_set = generate_data([0, 0], N)
  q_set = generate_data([10, 10], N)

  p_set = add_labels(p_set, -1.0)
  q_set = add_labels(q_set, 1.0)

  p_train, p_test = divide_dataset(p_set)
  q_train, q_test = divide_dataset(q_set)

  training_set = np.concatenate((p_train, q_train))
  test_set = np.concatenate((p_test, q_test))

  np.random.shuffle(training_set)
  np.random.shuffle(test_set)

  return training_set, test_set

def get_input(instance):
  """Extracts the input vector x and corresponding output label y

  This function also prepends 1 to input vector x for the bias

  Parameters:
    instance: The training instance

  Returns:
    The input x and label y
  """

  # prepend 1 to x for the bias
  x = np.insert(instance[1:], 0, 1)
  y = instance[0]
  return x, y

def get_label(x, w):
  """Determines the label given the input

  Parameters:
    x: The input vector
    w: The weight vector

  Returns
    The output label
  """

  return 1 if np.dot(w, x) >= 0 else -1

def classify(training_set, maxitercnt = 10000):
  """Implements the pocket algorithm for perceptron learning

  Parameters:
    training_set: The training set
    maxitercnt: The maximum number of iterations. Defaults to 10000

  Returns:
    The final weights after training
  """

  # size of training set
  N = len(training_set)

  # number of features
  d = len(training_set[0][1:])

  # initialize counters
  n_v = n_w = 0

  # initialize weight vectors
  v = w = np.zeros(d + 1)

  # pocket algorithm
  for itercnt in range(0, maxitercnt):
    i = np.random.choice(N)
    instance = training_set[i]
    x, y = get_input(instance)
    y_hat = get_label(x, v)

    if y * y_hat > 0:
      n_v = n_v + 1
    else:
      if n_v > n_w:
        w, n_w = v, n_v
      v, n_v = v + y * x, 0

  return w

def predict(test_set, w):
  """Predicts the labels of the given dataset

  Parameters:
    test_set: The test set whose labels need to be predicted
    w: The weights that will be used for prediction

  Returns:
    The expected and predicted labels
  """

  expected = predicted = np.array([])

  for instance in test_set:
    x, y = get_input(instance)
    y_hat = get_label(x, w)

    expected = np.append(expected, y)
    predicted = np.append(predicted, y_hat)

  return expected, predicted

def sse(expected, predicted):
  """Measure the sum of squared error

  Parameters:
    expected: The actual labels
    predicted: The predicted labels

  Returns:
    The sum of squared error
  """

  return np.sum((expected - predicted) ** 2)

def error_rate(expected, predicted):
  """Calculate the error rate i.e. the number of misclassified samples

  Parameters:
    expected: The actual labels
    predicted: The predicted labels

  Returns:
    The error rate
  """

  N = len(expected)
  errors = np.count_nonzero(expected != predicted)

  print('Total # of errors over {} test samples: {}'.format(N, errors))
  return errors / N

def start():
  """Main
  """

  print('** START POCKET ALGORITHM **\n')
  print('Generating training and test sets...\n')
  training_set, test_set = create_dataset()

  print('Determining weights using pocket algorithm...\n')
  weights = classify(training_set)

  print('Testing on test set...\n')
  expected, predicted = predict(test_set, weights)

  print('Error rate: {}\n'.format(error_rate(expected, predicted)))
  print('Sum of Squared Errors: {}'.format(sse(expected, predicted)))

  print('** END POCKET ALGORITHM **\n')

# run when not imported
if __name__ == "__main__":
  start()





