# numpy 기본 실습 과제.py
import numpy  as np
import scipy.misc
import matplotlib.pyplot as plt
import time

# 1번
# 1. numpy로 5행 6열 2차원 배열을 임의로 만들고 아래 지시대로 출력해보세요
d = np.arange(30).reshape(5,6)
print(d)

print('-'*80)

# 1.1 데이터를 거꾸로 출력해보세요.
print(d[::-1,::-1])

print('-'*80)
# 1.2 마지막 열을 제외한 모든 열을 출력해보세요.
print(d[:,:-1])

print('-'*80)
# 1.3 전치(transpose) 행렬을 출력해보세요
print(d.T)


print('-'*80)
# 1.4 2차원을 1차원 배열의 형태로 변형하여 출력하세요
print(d.flatten())


# 2번
face = scipy.misc.face()   # face: read only
face01 = face.copy()        # face01: read and write 가능
face02 = face.copy()        # face02: read and write 가능
face03 = face.copy()        # face03: read and write 가능
face04 = face.copy()        # face04: read and write 가능

# 2.1  Red 색상을 모두 0 으로 변경하여 출력한다
face01[:,:,0] = 0
# face01[face[:,:,0] < 100 ] = 0

# 2.2  Green 색상을 모두 0 으로 변경하여 출력한다
face02[:,:,1] = 0
# face02[face[:,:,1] < 100 ] = 0

# 2.3  Blue 색상을 모두 0 으로 변경하여 출력한다
face03[:,:,2] = 0
# face03[face[:,:,2] < 100 ] = 0

# 2.4  Red, Green, Blue 색상 중  100보다 작은 경우
# 모두 0 으로 변경하여 출력한다
#
time_start = time.time()
face04[face < 100] = 0
time_end = time.time()
print('elapsed time :',time_end - time_start)

# 중첩된 for문 사용하기
# time_start = time.time()
# xmax = face.shape[0]
# ymax = face.shape[1]
# zmax = face.shape[2]
# for i in range(xmax):
#     for j in range(ymax):
#         for k in range(zmax):
#             if face[i][j][k] < 100:
#                 face04[i][j][k] = 0
#
# time_end = time.time()
# print('elapsed time :',time_end - time_start)

plt.subplot(221)
plt.imshow(face01)

plt.subplot(222)
plt.imshow(face02)

plt.subplot(223)
plt.imshow(face03)

plt.subplot(224)
plt.imshow(face04)
plt.show()

print('-'*80)
# 3. numpy를 사용하여 아래 두개의 행렬을 만들고 지시대로 출력해보세요
a = np.arange(16).reshape(4,4)
print(a)
b = a*2
# 3.1 두개의 행렬을 수평으로 합쳐 결과를 출력하세요
h = np.hstack((a,b))
print(h)
print('-'*80)
# 3.2 두개의 행렬을 수직으로 합쳐 결과를 출력하세요
v = np.vstack((a,b))
print(v)
print('-'*80)
# 3.3 두개의 행렬을 열로 합쳐 결과를 출력하세요
c = np.column_stack((a,b))
print(c)

print('-'*80)
# 3.4 두개의 행렬을 행으로 합쳐 결과를 출력하세요
c = np.row_stack((a, b))
print(c)

