# iris_softmax_multi_classification_lab_answer.py

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

species_list =['"setosa"','"versicolor"','"virginica"']

xy = np.loadtxt('iris.csv',delimiter=',',dtype=np.str,skiprows=1)
x_train = np.float32(xy[:35,1:-1])
x_train = np.append(x_train , np.float32(xy[50:85,1:-1]),0)
x_train = np.append(x_train , np.float32(xy[100:135,1:-1]),0) # [105,4]

y_train = xy[:35,[-1] ]
y_train = np.append(y_train, xy[50:85,[-1]],0)
y_train = np.append(y_train, xy[100:135,[-1]],0) # [105,1]
for i in range(105):
   y_train[i,-1] = np.int32(species_list.index(y_train[i,-1]))
print(y_train)

x_test = np.float32(xy[35:50,1:-1])
x_test = np.append(x_test , np.float32(xy[85:100,1:-1]),0)
x_test = np.append(x_test , np.float32(xy[135:,1:-1]),0) # [45,4]

y_test = xy[35:50,[-1] ]
y_test = np.append(y_test, xy[85:100,[-1]],0)
y_test = np.append(y_test, xy[135:,[-1]],0) # [45,1]
for i in range(45):
   y_test[i,-1] = np.int32(species_list.index(y_test[i,-1]))
print(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

nb_classes = 3

X = tf.placeholder(tf.float32,shape=[None,4])
Y = tf.placeholder(tf.int32,shape=[None,1])

Y_one_hot = tf.one_hot(Y,nb_classes)  # [None,1,3]
print(Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes]) # [None,3]
print(Y_one_hot)

W = tf.Variable(tf.random_normal([4,nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                             labels = Y_one_hot)
cost =  tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train =optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# accuracy computation
predict = tf.argmax(hypothesis,1)  # 0~6
correct_predict = tf.equal(predict,tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,
                                     dtype = tf.float32))

# start training
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X:x_train, Y:y_train})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Accuracy report
h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X: x_test,Y:y_test})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)


# predict : test model
pred = sess.run(predict, feed_dict = {X:x_test})

for p,y in zip(pred, y_test.flatten()):
    print("[{}] Prediction: {} / Real Y: {} : {}".format(p == int(y), p,\
                                                   int(y),species_list[p]))

print('-'*20)
pred0 = sess.run(predict, feed_dict = {X:[[5.3,3.7,1.5,0.2]]})
print(pred0,':',species_list[pred0[0]]) # ==> 0

pred1 = sess.run(predict, feed_dict = {X:[[6.2,2.9,4.3,1.3]]})
print(pred1,':',species_list[pred1[0]]) # ==> 1

pred2 = sess.run(predict, feed_dict = {X:[[5.9,3,5.1,1.8]]})
print(pred2,':',species_list[pred2[0]]) # ==> 2