# 파이썬 데이터 분석


### scipy 홈페이지
https://scipy.org/

### numpy 홈페이지
http://www.numpy.org

### pandas 홈페이지
https://pandas.pydata.org/


#### Pandas: Setting no. of max rows in ipython notebook
pd.set_option('display.max_rows', 500)



-----------------------------------------------------------------------
# merge()활용
employee = [{'empno':1, 'ename':'kim', 'dept':1},
            {'empno':2, 'ename':'lee', 'dept':2}, 
            {'empno':3, 'ename':'park', 'dept':1}, 
            {'empno':4, 'ename':'song', 'dept':3},
            {'empno':5, 'ename':'min', 'dept':2} ]

dept=[{'dept':1, 'deptname':'관리직'},
      {'dept':2, 'deptname':'영업직'},
      {'dept':3, 'deptname':'개발직'} ]

info =[{'empno':1, 'addr':'서울시','phone':'010-1111-1111'},
       {'empno':3, 'addr':'부산시','phone':'010-2222-2222'}, 
       {'empno':2, 'addr':'광주시','phone':'010-3333-3333'}, 
       {'empno':5, 'addr':'광주시','phone':'010-4444-4444'},
       {'empno':4, 'addr':'광주시','phone':'010-5555-5555'} ]
