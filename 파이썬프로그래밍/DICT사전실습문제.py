# DICT실습문제.py
#

srp = {'가위':'보','바위':'가위','보':'바위'}

# 1
print(list(srp.keys()))

# 2
print(list(srp.values()))

# 3
print(list(srp.items()))

# 4
print(srp['가위'])

# 5
# 파이선 스타일 방식
a = [x for x,y in srp.items() if y == '바위']
print(a[0])

# 전통적인 언어의 방식
for x,y in srp.items():
    if y == '바위':
        a = x
print('key =',a)

# 6
b = {'찌':'빠', '묵':'찌', '빠':'묵'}
srp.update(b)
print(srp)

# 7
print('보자기' in srp)

# 8

# 파이선 스타일 방식
#srp = {1: '보',2:'바위', 3:'가위', 4:'묵', 5:'찌', 6:"빠"}
srp2 = { y:x for x,y in srp.items() }
print(srp2)

# 전통적인 언어의 방식
srp2 = {}
for x,y in srp.items():
    srp2.update({y:x})

print('srp2 =',srp2)


