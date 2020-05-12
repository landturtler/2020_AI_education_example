# mnist_cnn.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

mnist = input_data.read_data_sets('Data/mnist/', one_hot=True)

learning_rate = 0.001
training_epoch = 15
batch_size = 100

keep_prob = tf.placeholder(tf.float32)

# 입력 데이터, X,Y
X =  tf.placeholder(tf.float32,shape=[None,784])  # 28*28 픽셀
X_img = tf.reshape(X,[-1,28,28,1])
Y =  tf.placeholder(tf.float32,shape=[None,10])  # class = 10

# Layer 1  :  X_img:(?,28,28,1)
# Filter:(3,3,1,32), strides:(1,1,1,1), padding ='SAME'
W1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01),name='weight1')
L1 = tf.nn.conv2d(X_img,W1,strides=[1,1,1,1],padding='SAME') # (?,28,28,32)

# (N - F)/stride + 1
# L1 : (28+2 - 3)/1 + 1 = 28,  (?,28,28,32)
L1 = tf.nn.relu(L1)   # (?,28,28,32)
# kernel size : (1,2,2,1), strides:(1,2,2,1), padding ='SAME'
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding ='SAME')
L1 = tf.nn.dropout(L1,keep_prob=keep_prob)
# L1 : (28+1 - 2)/2 + 1 = 14,  (?,14,14,32)
print(W1)
print(L1)

# Layer 2  :  Image : (?,14,14,32)
# Filter:(3,3,32,64), strides:(1,1,1,1), padding ='SAME'
W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01),name='weight2')
L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME') # (?,

# (N - F)/stride + 1
# L2 : (14+2 - 3)/1 + 1 = 14,  (?,14,14,64)
L2 = tf.nn.relu(L2)   # (?,14,14,64)
# kernel size : (1,2,2,1), strides:(1,2,2,1), padding ='SAME'
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding ='SAME')
L2 = tf.nn.dropout(L2,keep_prob=keep_prob)
# L1 : (14+1 - 2)/2 + 1 = 7,  (?,7,7,64)
print(W2)
print(L2)    # (?,7,7,64)

# Layer 3  :  Image : (?,7,7,64)
# Filter:(3,3,64,128), strides:(1,1,1,1), padding ='SAME'
W3 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01),name='weight3')
L3 = tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME') # (?,7,7,128)

# (N - F)/stride + 1
# L3 : (7+2 - 3)/1 + 1 = 7,  (?,7,7,128)
L3 = tf.nn.relu(L3)   # (?,7,7,128)
# kernel size : (1,2,2,1), strides:(1,2,2,1), padding ='SAME'
L3 = tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding ='SAME')
L3 = tf.nn.dropout(L3,keep_prob=keep_prob)
# L1 : (7+1 - 2)/2 + 1 = 7,  (?,4,4,128)
print(W2)
print(L2)    # (?,4,4,128)

L3_flat = tf.reshape(L3,[-1,4*4*128]) # (?,4*4*128) 2차원

# layer 4 : FC(Fully Connected) Layer
# (?,4*4*128) * (4*4*128,625) = (?,625)
W4 = tf.get_variable("W4", shape=[4*4*128,625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]),name='bias4')
L4 = tf.nn.relu(tf.matmul(L3_flat,W4) + b4)
L4 = tf.nn.dropout(L4,keep_prob=keep_prob)

# layer 5 : FC(Fully Connected) Layer
# (?,625) * (625, 10) = (?,10)
W5= tf.get_variable("W5", shape=[625,10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]),name='bias5')

# 예측 방정식 H(X)
logits = tf.matmul(L4,W5) + b5
hypothesis = tf.nn.softmax(logits)   # Activation Function

# 비용함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(\
    logits=logits,labels=Y))

# optimizer : 경사하강법
optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimzer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train  : 학습 시작
print('Learning stared. It takes a long time!!')
for epoch in range(training_epoch) : # 15회 반복  , epoch : batch를 사용하여 데이터를 모두 소진한 1회
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)  # 55000/100= 550
    for i in range(total_batch):  # 550회 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 55000 개에서 100개를 추출, 데이터 샘플링
        cost_value,_ = sess.run([cost,train],feed_dict={X:batch_xs,Y:batch_ys,
                                                        keep_prob:0.7})  # 30% dropout
        avg_cost += cost_value/total_batch
    print('Epoch:','%04d'%(epoch + 1),'cost:','{:9f}'.format(avg_cost))

print("Learninig Finished")


# 예측 및 정확도 측정
predict = tf.argmax(hypothesis,1)
correct_predict = tf.equal(predict,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32))
a = sess.run(accuracy, feed_dict={X:mnist.test.images,Y:mnist.test.labels,
                                  keep_prob : 1.0}) #  0% dropout,test 데이터로 검증
print('Accuaracy:',a)

# softmax 4 layers Accuaracy:  0.9837  , relu() 사용, AdamOptimizer, xavier_initializer()사용
                                        # dropout 사용(과적합을 방지)
# CNN사용 3 layers Accuaracy:  0.9818

# CNN사용 5 layers Accuaracy:  0.9943   # dropout 사용(과적합을 방지), 2개 FC

# 예측 결과 시각화 : matplotlib()
r = random.randint(0,mnist.test.num_examples - 1) # (0,10000 - 1)
print('random:',r,'Label:',sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print('Predict:',sess.run(predict,feed_dict={X:mnist.test.images[r:r+1],
                                             keep_prob: 1.0}))
plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
plt.show()
