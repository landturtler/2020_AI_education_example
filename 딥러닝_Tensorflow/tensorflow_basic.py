# tensorflow_basic.py
#
import tensorflow as tf

a = tf.constant(10)
b = tf.constant(20)
c = tf.add(a,b)
print(c)
sess = tf.Session()
print(sess.run(c))   # 30
sess.close()
print('-'*30)
# 상수
a = tf.constant('Hello')
sess = tf.Session()
print(sess.run(a))   # 30
sess.close()

# 변수 : Variable(), placeholder()
# (1) Variable() : 초기 값이 설정, 주로 텐서플로 내부에서 연산시 변경되는 변수, weight 와 bias
a = tf.Variable(100)
b = tf.Variable(200)
c = tf.add(a,b)
print(c)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(c))   # 300     (지연 실행, lazy evaluation)
sess.close()

# (2) placeholder() : 초기 값이 정해지지 않고 sess.run() 실행시 변수를 입력
# feed_dict = {X:[,,,]  , Y:[,,,] }
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
# c = a + b
c = tf.add(a,b)
# print(c)
sess = tf.Session()
print(sess.run(c,feed_dict= {a:100, b:200}))   # 300
print(sess.run(c,feed_dict= {a:10, b:20}))   # 30
print(sess.run(c,feed_dict= {a:[1,2,3], b:[10,20,30]}))   # [11 22 33]

sess.close()

