# SET실습문제.py

# 1
a = [1,2,3,4]
s1 = set(a)
print(s1)

b = "aabbccddeeff"
s2 = set(b)
print(s2)

# 2
s1.update({'a','b','c'})
print(s1)

# 3
s2.update({1,2})
print(s2)

# 4
s = s1 & s2
print(s)

s = s1.intersection(s2)
print(s)

# 5
s = s1 | s2
print(s)

s = s1.union(s2)
print(s)

# 6
s = s1 - s2
print(s)

s = s1.difference(s2)
print(s)

# 7
s = s2.difference(s1)
print(s)

# 8
s1.remove(1)
print(s1)

# 9
s = s1.symmetric_difference(s2)
print(s)

s = (s1- s2) | (s2 - s1)
print(s)

