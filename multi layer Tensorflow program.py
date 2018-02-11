
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
import time as T
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

x_test, y_test = data.test.next_batch(10000)


# To make debugging easy add seed
seed = 1298
rng = np.random.RandomState(seed)

#load training dataset
x_train, y_train = data.train.next_batch(110000)
#train = pd.read_csv('train.csv')

#set all variables

# number of neurons in each layer

input_num_units = 28*28
hidden_num_units = 300
output_num_units = 10
input_num_units

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])


# define model learning rate
learning_rate = 0.001


weights = {
    'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units, hidden_num_units], seed=seed)),
    'hidden3': tf.Variable(tf.random_normal([hidden_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'hidden3': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}


# define layers


hidden_layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden_layer1 = tf.nn.relu(hidden_layer1)

hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])
hidden_layer2 = tf.nn.relu(hidden_layer2)

hidden_layer3 = tf.add(tf.matmul(hidden_layer2, weights['hidden3']), biases['hidden3'])
hidden_layer3 = tf.nn.relu(hidden_layer3)

output_layer = tf.matmul(hidden_layer3, weights['output']) + biases['output']

#to calculate cost of our neural network; cost= summation of all errors

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,labels = y))

# set the optimizer, i.e. our backpropogation algorithm.
# using Adam optimiser to reduce error by updating weights

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize all the variables 

init = tf.global_variables_initializer()

#start session
sess=tf.Session()

# create initialized variables
sess.run(init)


# this is where the magic happenes #training

t=T.clock()
for i in range(11000):
    huyt=x_batch[i*10:(i+1)*10-1]
    sdws=y_true_batch[i*10:(i+1)*10-1]
    _, c = sess.run([optimizer,cost],feed_dict={x:huyt,y:sdws})
    if i%1000==0:
        print c/1000,
print "time to tain: ",T.clock()-t


# testing step

t=T.clock()

predict = tf.argmax(output_layer, 1)
pred = sess.run(predict,{x: x_test})

print "time to test: ",T.clock()-t

#calculating error %
j=0
jui=[]
coun=10000
for i in xrange(coun):
    if pred[i]!=np.argmax(y_test[i]):
        j+=1;jui.append(i)
err=j*100.0/coun
print "error: ",err," count: ",len(jui)

#
get_ipython().magic(u'matplotlib inline')
img_shape=(28,28)
iou=4 #give an index to see the prediction of the model

plt.imshow(x_test[iou].reshape(img_shape), cmap='binary')

print 'predicted: ',pred[iou],"truth: ",np.argmax(y_test[iou])


# the end
