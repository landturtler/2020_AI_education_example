# Boston_Dataset_Linear Regression.py
import tensorflow as tf
import numpy as np
import csv
tf.set_random_seed(777)

boston_train = np.loadtxt('boston_train.csv', delimiter=',', dtype=np.float32, skiprows=1)
boston_test = np.loadtxt('boston_test.csv', delimiter=',', dtype=np.float32, skiprows=1)

x_data = boston_train[:,:9]
y_data = boston_train[:,[-1]]
# print(x_data.shape)
# print(y_data.shape)


X = tf.placeholder(tf.float32, shape=[None, 9])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([9, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(1000001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost,W,b,train],
                 feed_dict = {X:x_data, Y:y_data})
    if step %10000 == 0:
        print(step,cost_val)#, W_val, b_val)


# test model
x_test = boston_test[:,:9]
y_test = boston_test[:,[-1]]
print('y_test : ',y_test,'\ny_test :',\
      sess.run(hypothesis,feed_dict={X:x_test}),\
      '\ncost :',y_test-sess.run(hypothesis,\
                                  feed_dict={X:x_test}))
