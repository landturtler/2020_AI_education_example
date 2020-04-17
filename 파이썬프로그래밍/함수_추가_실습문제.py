# 함수실습문제.py

#
from collections import namedtuple


def printList(title, numberPerLine, prnList):
    print('\n{} :'.format(title))
    count = 0
    for x in prnList:
        count = count + 1
        print('{:>6}'.format(x), end='')
        if count % numberPerLine == 0:
            print()
            count = 0


print('{0:=^50}'.format(' 1번 '))


def calc_average(number1, number2):
    total = number1 + number2
    average = total / 2
    return average


number1 = int(input('Input number1 : '))
while number1 != -1:
    number2 = int(input('Input number2 : '))

    if number2 != 0:
        result = calc_average(number1, number2)
        prn_format = '( {:^5} + {:^5} ) / 2 = {:6.2f}'.format(number1, number2, result)
        print(prn_format)
    else:
        print('Error : number2 => {:<5} == 0'.format(number2))

    print()
    number1 = int(input('Input number1 : '))
print('{:=^50}\n'.format(' End Program 1번 '))

print('{0:=^50}'.format(' 2번 '))


def calc_Max_Min(number_list):
    return max(number_list), min(number_list)


number_list = []

number = int(input('Input number1 : '))
while number != -1:
    number_list.append(number)

    number = int(input('Input number1 : '))

max_number, min_number = calc_Max_Min(number_list)

printList('Input number list :', 8, number_list)
print('\n')
print('Max number : {:<6}Min number : {:<6}'.format(max_number, min_number))
print('{:=^50}\n'.format(' End Program 2번 '))

print('{0:=^50}'.format(' 3번 '))


def totalOfInteger(start, end):
    l = [x for x in range(start, end + 1)]
    result = sum(l)
    return result, l


while True:
    start = int(input('Input start number( quit = start > end ) : '))
    end = int(input('Input end number( quit = start > end ) : '))

    if start < end:
        result, l = totalOfInteger(start, end)
        printList('number list', 10, l)
        print('\nsum = {:<6}\n'.format(result))
    else:
        break

print('{:=^50}\n'.format(' End Program 3번 '))

print('{0:=^50}'.format(' 4번 '))


def operationString(stringList):
    character3List = [x[:3] for x in stringList]
    return character3List


stringList = []
while True:
    string = input('Input string : ')
    if string != 'end':
        stringList.append(string)
    else:
        break

count = 1
for x in operationString(stringList):
    print('{0:>3} : {1:<30}'.format(count, x))
    count += 1

print('{:=^50}\n'.format(' End Program 4번 '))

print('{0:=^50}'.format(' 5번 '))


def myrange(*args):
    length_args = len(args)

    if length_args == 1:
        start = 0
        end = args[0]
        interval = 1
    elif length_args == 2:
        start = args[0]
        end = args[1]
        interval = 1
    else:
        start = args[0]
        end = args[1]
        interval = args[2]

    # loop_list = [x for x in range(start, end, interval)]

    loop_list = []
    while True:
        x = start
        loop_list.append(x)
        start = start + interval
        if start >= end :
            break

    result = tuple(loop_list)
    return result


t = myrange(10)
printList('myrange( 10 ) result ', 10, t)

t = myrange(1, 10)
printList('myrange( 1, 10 ) result ', 10, t)

t = myrange(1, 10, 2)
printList('\nmyrange( 1, 10, 2 ) result ', 10, t)

print('\nloop -> myrange( 20 )')
for x in myrange(20):
    print('{:<3}'.format(x), end=' ')
print('\n{:=^50}\n'.format(' End Program 5번 '))

print('{0:=^50}'.format(' 6번  '))
menu = '''
1. add
2. subtract
3. multiply
4. divide
0. end
select : '''


def add(number1, number2):
    return number1 + number2


def subtract(number1, number2):
    return number1 - number2


def multiply(number1, number2):
    return number1 * number2


def divide(number1, number2):
    return number1 / number2


op_func = {1: add, 2: subtract, 3: multiply, 4: divide}
op_char = {1: '+', 2: '-', 3: '*', 4: '/'}

while True:
    print(menu, end=' ')
    select_menu = int(input())

    if select_menu == 0:
        break;
    elif 1 <= select_menu <= 4:
        number1 = int(input('Input number1 : '))
        number2 = int(input('Input number2 : '))
        result = op_func[select_menu](number1, number2)
        print('{0:<5} {1:^3} {2:<5} = {3:<6.2f}'. \
              format(number1, op_char[select_menu], number2, result))

print('\n{:=^50}\n'.format(' End Program 6번 '))

print('{0:=^50}'.format(' 7번  '))

Student = namedtuple('Student', 'name, subject1, subject2, subject3, total, average, grade')

student_list = []
MAX = 10
SUBJECT = 3


def calcScore(subject):
    total = 0
    for x in subject:
        total = total + x

    average = total / SUBJECT

    if average >= 90:
        grade = 'Excellent'
    elif average <= 50:
        grade = 'Fail'
    else:
        grade = ' '

    return total, average, grade


def inputStudentInfo(student_list):
    count = 0
    name = input('Input name : ')
    while name != 'end' and count < MAX:
        count += 1
        subject = []
        for x in range(SUBJECT):
            input_subject = int(input('Input subject' + str(x + 1) + ': '))
            subject.append(input_subject)
        total, average, grade = calcScore(subject)
        student = Student(name, subject[0], subject[1], subject[2], total, average, grade)
        student_list.append(student)

        print('\n')
        name = input('Input name : ')

    return


def printScoreTable(student_list):
    for x in student_list:
        print('{0:<10} {1:<3} {2:<3} {3:<3} {4:<5} {5:6.2f} {6:<10}'. \
              format(x.name, x.subject1, x.subject2, x.subject3, x.total, x.average, x.grade))

    return


inputStudentInfo(student_list)
print('\n')
printScoreTable(student_list)

print('\n{:=^50}\n'.format(' End Program 7번 '))
