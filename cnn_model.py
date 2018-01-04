from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from cifarnet import *
from data_utils import load_CIFAR10
import time

slim = tf.contrib.slim

import pickle
beginTime = time.time()

# Parameter definitions
weight_decay = 0.0
batch_size = 100
learning_rate = 0.005
max_steps = 100000


print('weight_decay: {}'.format(weight_decay))

def load_data():
  '''load all CIFAR-10 data'''

  x_train, y_train, x_test, y_test = load_CIFAR10('cifar-10-batches-py')

  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck']

  # Normalize Data
  mean_image = np.mean(x_train, axis=0)
  x_train -= mean_image
  x_test -= mean_image

  data_dict = {
    'images_train': x_train,
    'labels_train': y_train,
    'images_test': x_test,
    'labels_test': y_test,
    'classes': classes
  }
  return data_dict
# Prepare data
data_sets = load_data()

# -----------------------------------------------------------------------------
# Prepare the TensorFlow graph
# (We're only defining the graph here, no actual calculations taking place)
# -----------------------------------------------------------------------------

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define the classifier's result
with slim.arg_scope(cifarnet_arg_scope(weight_decay)):
  logits, _ = cifarnet(images_placeholder, is_training=True)

# Define the loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
  labels=labels_placeholder))

# Define the training operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------
with tf.Session() as sess:
  # Initialize variables
  sess.run(tf.global_variables_initializer())

  # Repeat max_steps times
  for i in range(max_steps):

    # Generate input data batch
    indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
    images_batch = data_sets['images_train'][indices]
    labels_batch = data_sets['labels_train'][indices]

    # Periodically print out the model's current accuracy
    if i % 1000 == 0:
      train_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: images_batch, labels_placeholder: labels_batch})
      print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

    if i % 10000 == 0:
      test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: data_sets['images_test'],
        labels_placeholder: data_sets['labels_test']})
      print('Test accuracy {:g}'.format(test_accuracy))

    # Perform a single training step
    sess.run(train_step, feed_dict={images_placeholder: images_batch,
      labels_placeholder: labels_batch})

  save_path = 'output/cifarnet'
  saver.save(sess, save_path)

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
