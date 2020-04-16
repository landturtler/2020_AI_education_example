# 튜플실습문제.py

a=('a1','a2','a3','a4')

b=('b1','b2','b3','b4')

# 1
q, w, e, r = a   # 언패킹
print(q,w,e,r)

# 2
c = a + b       # + 연산
print(c)

# 3
print(c[2])     # 인덱싱

# 4
print(c[5:])    # 슬라이싱

# 5
print(c[:3])    # 슬라이싱

# 6
del a[3]
# TypeError: 'tuple' object doesn't support item deletion

#7
c[4] = 'c1'
# TypeError: 'tuple' object does not support item assignment