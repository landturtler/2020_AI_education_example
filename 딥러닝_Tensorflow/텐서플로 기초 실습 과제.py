# 텐서플로 기초 실습 과제.py
# placeholder 사용법

import tensorflow as tf
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# [과제 1] placeholder 를 사용하여 덧셈 연산을 텐서플로로 구현하세요

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
add_ab = tf.add(a,b)
sess = tf.Session()
print(sess.run(add_ab, feed_dict= {a : [10,20,30,40,50],
                                   b : [1,2,3,4,5]}))
sess.close()


# [과제 2] 다음 함수를 텐서플로로 구현하세요  h(x) = 3*x - 5
x = tf.placeholder(tf.int32)
hx = tf.subtract(tf.multiply(3,x),5)
sess = tf.Session()
print(sess.run(hx, feed_dict= {x:[1,2,3]}))
sess.close()

# [과제 3] 텐서플로를 사용하여 1 부터 100 까지의 합과 평균을 출력하는 코드를 구현하세요
mylist = list(range(1,101))
x = tf.placeholder(tf.int32)
mysum = tf.reduce_sum(x)
mymean = tf.reduce_mean(x)
sess = tf.Session()
print(sess.run([mysum,mymean], feed_dict= {x:mylist}))
sess.close()