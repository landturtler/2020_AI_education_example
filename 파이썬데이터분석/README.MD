# 파이썬 데이터 분석


### scipy 홈페이지
https://scipy.org/

### numpy 홈페이지
http://www.numpy.org

### pandas 홈페이지
https://pandas.pydata.org/

### 인공지능을 위한 기초수학
http://matrix.skku.ac.kr/math4ai/

#### Pandas: Setting no. of max rows in ipython notebook
pd.set_option('display.max_rows', 500)

#### 그래프를 노트북 화면 내부에 출력
 %matplotlib inline 
#### 그래프를 노트북 화면 외부에 출력 
 %matplotlib qt      
 
####
####

=====================================================
- merge() 예제 리스트 데이터셋

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
       


####
####

=================================================
- grouby() 와 pivot_table() 예제 데이터셋

from numpy.random import seed

from numpy.random import rand

from numpy.random import randint

seed(42)

df = pd.DataFrame({
   'Weather' : ['cold', 'hot', 'cold', 'hot',
   'cold', 'hot', 'cold'],
   
   'Food' : ['soup', 'soup', 'icecream', 'chocolate',
   'icecream', 'icecream', 'soup'],
   
   'Price' : 10 * rand(7), 'Number' : randint(1, 9, 7)})
