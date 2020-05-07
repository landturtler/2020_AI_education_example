# LinearRegression.py

import tensorflow as tf
tf.set_random_seed(777)  # random 값이 예측 정확도에 영향을 주지 않도록 seed를 고정

# 학습 데이터 : X 와 Y 데이터
x_train = [1,2,3]
y_train = [1,2,3]

# 변수 초기화 : weight, bias
# tf.random_normal([1]) : 난수를 1개 생성
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# hypothesis 예측 함수(방정식)  , H(x) = W * X + b
hypothesis =  x_train * W + b
# hypothesis =  tf.add(tf.multiply(x_train,W), b)

# 비용함수 : (Hx - y)^2 의 평균
# tf.square() : 제곱
# tf.reduce_mean() : 함의 평균
cost =  tf.reduce_mean(tf.square(hypothesis - y_train))

# 경사 하강법
# learning_rate (학습율) 을 0.01로 설정하여  optimizer객체를 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# session 객체를 생성
sess = tf.Session()

# Variable 초기화
sess.run(tf.global_variables_initializer())

# 학습
for step in range(10001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W),sess.run(b) )

print('weight =',sess.run(W))
print('bias =',sess.run(b))

# 예측
predict = 5 * W + b
print('x=5 , H(x)=',sess.run(predict))
predict = 11 * W + b
print('x=11 , H(x)=',sess.run(predict))
