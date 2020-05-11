# mnist_example.py
# MNIST(Modified National Institute of Standard Technology database)
# label : 0 ~ 9 , 손글씨체 이미지, 28*28 , gray scale, 784 bytes

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

mnist = input_data.read_data_sets('Data/mnist/', one_hot=True)
# t10k-images-idx3-ubyte.gz  : 이미지파일
# t10k-labels-idx1-ubyte.gz  :  label(0~9)

def show_one_image():
    train = mnist.train.images
    test = mnist.test.images
    # print(type(mnist))
    print(type(train),train.shape)              # (55000,784)
    print(type(test),test.shape)                # (10000,784)
    print(type(mnist.train.labels),mnist.train.labels.shape)  # (55000,10)
    print(type(mnist.test.labels),mnist.test.labels.shape)    # (10000,10)

    image = mnist.train.images[0].reshape(28,28)
    print(mnist.train.labels[0])  # [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] : 7

    plt.imshow(image,cmap='Greys')
    plt.show()

# show_one_image()

# softmax사용 : class = 10 , multi-nomial classification


# 파라메터 값
learning_rate = 0.001
training_epoch = 15
batch_size = 100

keep_probe = tf.placeholder(tf.float32)

# 입력 데이터, X,Y
X =  tf.placeholder(tf.float32,shape=[None,784])  # 28*28 픽셀
Y =  tf.placeholder(tf.float32,shape=[None,10])  # class = 10
print(X)

# layer 1
# W1 = tf.Variable(tf.random_normal([784,512],name='weight1'))
W1 = tf.get_variable("W1", shape=[784,512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]),name='bias1')
# L1 = tf.sigmoid(tf.matmul(X,W1) + b1)
L1 = tf.nn.relu(tf.matmul(X,W1) + b1)
L1 = tf.nn.dropout(L1,keep_prob=keep_probe)

# layer 2
# W2 = tf.Variable(tf.random_normal([512,512],name='weight2'))
W2 = tf.get_variable("W2", shape=[512,512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]),name='bias2')
# L2 = tf.sigmoid(tf.matmul(L1,W2) + b2)
L2 = tf.nn.relu(tf.matmul(L1,W2) + b2)
L2 = tf.nn.dropout(L2,keep_prob=keep_probe)

# layer 3
# W3 = tf.Variable(tf.random_normal([512,512],name='weight3'))
W3 = tf.get_variable("W3", shape=[512,512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]),name='bias3')
# L3 = tf.sigmoid(tf.matmul(L2,W3) + b3)
L3 = tf.nn.relu(tf.matmul(L2,W3) + b3)
L3 = tf.nn.dropout(L3,keep_prob=keep_probe)

# layer 4
# W4 = tf.Variable(tf.random_normal([512,10],name='weight5'))
W4 = tf.get_variable("W4", shape=[512,10],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10]),name='bias5')

logits = tf.matmul(L3,W4) + b4
hypothesis = tf.nn.softmax(logits)

# 비용함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(\
    logits=logits,labels=Y))

# optimizer : 경사하강법
# optimzer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimzer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습 : train
print('Start Learning!!')
for epoch in range(training_epoch) : # 15회 반복  , epoch : batch를 사용하여 데이터를 모두 소진한 1회
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)  # 55000/100= 550
    for i in range(total_batch):  # 550회 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 55000 개에서 100개를 추출, 데이터 샘플링
        cost_value,_ = sess.run([cost,train],feed_dict={X:batch_xs,Y:batch_ys,
                                                        keep_probe:0.7})  # 30% dropout
        avg_cost += cost_value/total_batch
    print('Epoch:','%04d'%(epoch + 1),'cost:','{:9f}'.format(avg_cost))

print("Learninig Finished")

# 예측 및 정확도 측정
predict = tf.argmax(hypothesis,1)
correct_predict = tf.equal(predict,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32))

a = sess.run(accuracy, feed_dict={X:mnist.test.images,Y:mnist.test.labels,
                                  keep_probe : 1.0})  # 0% dropout,  test 데이터로 검증
print('Accuaracy:',a)

# softmax 1 layers Accuaracy:  0.7941
# softmax 3 layers Accuaracy:  0.8023  , sigmoid() 사용
# softmax 5 layers Accuaracy:  0.7293  , sigmoid() 사용
# softmax 4 layers Accuaracy:  0.9301  , relu() 사용, GradientDescentOptimizer
# softmax 4 layers Accuaracy:  0.9582  , relu() 사용, AdamOptimizer
# softmax 4 layers Accuaracy:  0.9766  , relu() 사용, AdamOptimizer, xavier_initializer()사용
# softmax 5 layers Accuaracy:  0.9826  , relu() 사용, AdamOptimizer, xavier_initializer()사용
                                        # dropout 사용(과적합을 방지)
# softmax 최대 정확도
# softmax 4 layers Accuaracy:  0.9837  , relu() 사용, AdamOptimizer, xavier_initializer()사용
                                        # dropout 사용(과적합을 방지)


# 예측 결과 시각화 : matplotlib()
r = random.randint(0,mnist.test.num_examples - 1) # (0,10000 - 1)
print('random:',r,'Label:',sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print('Predict:',sess.run(predict,feed_dict={X:mnist.test.images[r:r+1],
                                             keep_probe : 1.0}))
plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
plt.show()
