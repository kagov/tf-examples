import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Works only in IPython,
# in a normal file you have to save the image to filesystem and open it
#%matplotlib inline#

# Load the data, It contains the population
# profit in millions
# Convert the data as a numpy array and separate the features and labels
def load_data():
    data = pd.read_csv('data.txt')
    x= data.as_matrix(columns=None)
    x_batch = x[:96 ,:1 ]
    print (x_batch.shape)
    y_batch = x[:96,1:]
    print (y_batch.shape)
    return x_batch,y_batch

# Perform Linear regression
# Create placeholders for features and labels
# Initialize the weights and the bias
# Find the predictions (y = mx + c)
# Calculate the error or loss (Mean squared error)
# Return the placeholders for x and y along with the predictions and loss
def linear_regression():
    x = tf.placeholder(tf.float32, shape=(None, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    with tf.variable_scope('lreg') as scope:
        # w = tf.Variable(np.random.normal(),name = 'w')
        w = tf.Variable(np.zeros((2,), dtype=np.float32), name='w')
        b = tf.Variable(np.ones((2,), dtype=np.float32), name='bias')
        y_pred = tf.multiply(w, x) + b
        loss = tf.reduce_mean(tf.square(y_pred - y))

    return x, y, y_pred, loss

# Obtain a TF Session and train
# Gradient Descent is used to update the weights and reduce the error
# Learning rate is 0.01
# Run the algorithm for 1500 iterations and print the loss
# Notice that the loss decreases over iterations
# Calculate the predicted labels (y_pred_batch)
# Plot the input data and the predicted output
# Save the plot
def run():
    x_batch, y_batch = load_data()
    x, y, y_pred, loss = linear_regression()
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    feed_dict = {x: x_batch, y: y_batch}
    for i in range(1500):
        loss_val, _ = sess.run([loss, optimizer], feed_dict)
        if (i % 150 == 0):
            print('loss: ', loss_val)

    y_pred_batch = sess.run(y_pred, {x: x_batch})
    plt.scatter(x_batch, y_batch)
    plt.plot(x_batch, y_pred_batch)
    plt.savefig('plot.png')

if __name__ == '__main__':
    run()