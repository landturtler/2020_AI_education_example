# 문자열실습문제.py

# 1번 답안
print( '{0:=^50}'.format( '1' ) )
s = 'hong gil dong201912121623210'

name = s[ :s.rfind( 'g' ) + 1 ]
birthday_index = s.rfind( 'g' ) + 1
birthday = s[ birthday_index: birthday_index + 9 ]
idnumber = s[ s.rfind( 'g' ) + 1: ]

print( 'Name : ' + name )
print( 'Birthday : ' + birthday[ :4 ] + '/' + birthday[ 4:6 ] + '/' + birthday[ 6:8 ] )
print( 'ID Number : ' + idnumber[ :8 ] + '-' + idnumber[ 8: ] )

# 2번 답안
print( '{0:=^50}'.format( '2' ) )
s = 'PythonProgramming'

print( s[ s.find( 'Programming' ): ] + s[ :s.find( 'Programming' ) ] )

# 3번 답안
print( '{0:=^50}'.format( '3' ) )

s = 'hello world'

print( 'hi' + ' ' + s[ s.find( 'world' ): ] )
