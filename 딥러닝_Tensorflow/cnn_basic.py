# cnn_basic.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
#
# a = tf.constant(10)
# b = tf.constant(20)
# add_node = tf.add(a,b)
#
# print(add_node.eval())
# print(sess.run(add_node))

image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)
print(image)
print(image.shape)  # (1,3,3,1)
# plt.imshow(image.reshape(3,3),cmap='Greys')
# print(image.reshape(3,3))
# plt.show()

# CNN(Convolutional Neural Network)
# conv2d layer   , filter (2,2,1,1)
#  출력 이미지 = (N - F)/stride  + 1

# image (1,3,3,1)
# 1 : 이미지의 갯수
# 3 : 가로 픽셀수
# 3 : 세로 픽셀 수
# 1 : color, Greyscale = 1

# filter (2,2,1,1)
# 2 : 가로 갯수
# 2 : 세로 갯수
# 1 : color
# 1 : filter의 갯수
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
conv2d = tf.nn.conv2d(image,weight,stride=[1,1,1,1])