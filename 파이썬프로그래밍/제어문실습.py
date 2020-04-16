# 제어문실습.py

# 1번
print('{0:=^50}'.format('1-1'))
for x in range(1, 101):
    print('{:4}'.format(x), end='')
    if x % 10 == 0:
        print()

print('{0:=^50}'.format('1-2'))
l = [x for x in range(1, 101)]
for x in l:
    print('{:4}'.format(x), end='')
    if x % 10 == 0:
        print()

# 2번
print('{0:=^50}'.format('2'))
max_number = int(input('Input max number : '))

total = 0
for x in range(1, max_number + 1):
    total = total + x

print('1 ~ {0:^6} = {1:<8}'.format(max_number, total))

# 3번
print('{0:=^50}'.format('3'))
max_number = int(input('Input max number : '))

even_list = []
odd_list = []
for x in range(1, max_number + 1):
    if x % 2 == 0:
        even_list.append(x)
    else:
        odd_list.append(x)

print('even number : ', even_list)
print('1 ~ {0:^6} = {1:<8d}\n'.format(max_number, sum(even_list)))

print('odd number : ', odd_list)
print('1 ~ {0:^6} = {1:<8d}'.format(max_number, sum(odd_list)))

# 4번
print('{0:=^50}'.format('4'))
max_number = int(input('Input max number : '))

Excluding_Multiple_of_3_5 = []

for x in range(1, max_number + 1):
    if x % 3 != 0 and x % 5 != 0:
        Excluding_Multiple_of_3_5.append(x)

print('Excluding Multiple of 3 and 5 : ', Excluding_Multiple_of_3_5)
print('sum = {0:<6}'.format(sum(Excluding_Multiple_of_3_5)))

# 5번
print('{0:=^50}'.format('5-1'))
for x in range(2, 10):
    for y in range(1, 10):
        print('{:3}'.format(x * y), end='')
    print()

print('{0:=^50}'.format('5-2'))
multiple_table = [x * y for x in range(2, 10) for y in range(1, 10)]

count = 0
for x in range(8 * 9):
    count = count + 1
    print('{:3}'.format(multiple_table[x]), end='')
    if count % 9 == 0:
        print()
        count = 0

print('{0:=^50}'.format('5-3'))
multiple_table2 = [x * y for x in range(2, 10) for y in range(1, 10)]

start = 0
for x in range(9, 81, 9):
    print('{0[0]:3}{0[1]:3}{0[2]:3}{0[3]:3}{0[4]:3}{0[5]:3}{0[6]:3}{0[7]:3}{0[8]:3}' \
          .format(multiple_table2[start:x]))
    start = start + 9

# 6번
print('{0:=^50}'.format('6'))
total_list = [0, 0, 0, 0]
total_title = ('positive', 'negative', 'even', 'odd')

while True:
    number = int(input('Input number : '))
    if number == -999:
        break

    if number != 0:
        if number > 0:
            total_list[0] = total_list[0] + 1
            if number % 2 == 0:
                total_list[2] = total_list[2] + 1
            else:
                total_list[3] = total_list[3] + 1
        else:
            total_list[1] = total_list[1] + 1
    else:
        print('error : input not {}'.format(number))

print()
for x in range(4):
    print('{0:<10} : {1:<5}'.format(total_title[x], total_list[x]))

# 7번
print('{0:=^50}'.format('7'))
op = {1: '+', 2: '-', 3: '*', 4: '/'}

while True:
    number1 = input('Input number1 : ')
    number2 = input('Input number2 : ')
    op_select = int(input('Input operator( 1:+, 2:-, 3:*, 4:/, 0:end ) : '))

    if op_select == 0:
        break;

    result = eval(number1 + op[op_select] + number2)

    print('number1 : {0:^8.2}'.format(number1))
    print('number2 : {0:^8.2}'.format(number2))
    print('{0:^6} {2:^3} {1:^6} = {3:<.2f}\n'.format(number1, number2, op[op_select], result))

# 8번
print('{0:=^50}'.format('8'))
from collections import namedtuple

Student = namedtuple('Student', 'name, subject1, subject2, subject3, total, average, grade')

student_list = []
MAX = 10
SUBJECT = 3
count = 0

name = input('Input name : ')
while name != 'end' and count < MAX:

    count = count + 1
    subject = []
    for x in range(SUBJECT):
        input_subject = int(input('Input subject' + str(x) + ':'))
        subject.append(input_subject)
    total = sum(subject)
    average = total / SUBJECT
    if average >= 90:
        grade = 'Excellent'
    elif average <= 50:
        grade = 'Fail'
    else:
        grade = ' '

    student = Student(name, subject[0], subject[1], subject[2], total, average, grade)
    student_list.append(student)

    name = input('Input name : ')

print()
for x in student_list:
    print('{0:<10} {1:<3} {2:<3} {3:<3} {4:<5} {5:6.2f} {6:<10}'. \
          format(x.name, x.subject1, x.subject2, x.subject3, x.total, x.average, x.grade))

