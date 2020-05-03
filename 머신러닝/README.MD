### 파이썬 활용 머신러닝
#### 참고 사이트


- Kaggle

    https://www.kaggle.com/


- UCI Machine Learning Repository (머신러닝 데이터)
    
    https://archive.ics.uci.edu/ml/index.php
    
    
- 데이콘 (경진대회)   
    https://dacon.io/

- 조대협의 블로그
    https://bcho.tistory.com/tag/F1%20score


- graphviz    
    https://graphviz.gitlab.io/_pages/Download/Download_windows.html

<br>

#### * 캐글 신용카드 사기 검출(Credit Card Fraud Detection) 데이터 셋
#### https://www.kaggle.com/mlg-ulb/creditcardfraud
#### 에서 구글계정 등으로 가입하여 로그인 하고 우측상단의  'Download (144 MB)'을 눌러 
#### creditcardfraud.zip 파일을 다운받아서 압축을 해제하여 소스 디렉토리에 옮겨 놓는다 
#### creditcard.csv(약 143 MB)  ==> LMS에 올려져 있습니다
<br>

#### 딥러닝 강좌  

- 모두를 위한 머신러닝과 딥러닝의 강의(김성훈교수님)

    https://hunkim.github.io/ml/
##### ====================================================================================
#### 
### 인공지능의 역사 : 신경망(DNN,Deep Neural Net) 역사
##### ====================================================================================
#### [1] 퍼셉트론 : 다수의 신호를 입력받아서 하나의 신호로 출력, 신경망의 기원이 되는 알고리즘,
Frank Rosenblatt(1957,Cornell Aeronautical Lab 알고리즘 개발)
#### [2] XOR Problem : Marvin Minsky(1969,MIT AI Lab founder)
퍼셉트론은 AND, OR, NAND 같은 선형문제는 풀수 있을지 모르지만, XOR같은 비선형 문제를 풀수가 없다--> 인공지능의 1차 겨울(빙하기)
#### [3] Backpropagation : 1986년 제프리 힌튼(Geoffrey Hinton)
샘플에 대한 신경망의 오차를 다시 출력층에서부터 입력층으로 거꾸로 전파시켜 각 층의 가중치(weight)를 계산하는 방법. 이를 통해 weight와 bias를 알맞게 학습할 수 있다
#### [4] 두 번째 겨울(빙하기) : 1990년대 후반
* (1) Vanishing Gradient : 신경망의 깊이가 깊어질수록 원하는 결과를 얻을 수 없다
* (2) 신경망 학습을 위한 파라미터 값의 최적화에 대한 이론적인 근거가 없었다.
#### [5] Deep의 출현 : 2006년 , Geoffrey Hinton, "A fast learning algorithm for deep belief nets" 라는 논문
weight의 초기값을 제대로 설정하면 깊은 신경망학습이 가능하다 neural networke 대신 Deep Network, Deep Learning 이라는 용어가 사용되기 시작했다.


##### ====================================================================================
####
### 머신 러닝 개념
##### ====================================================================================
#### 인공지능(AI) 분류
##### [1] 규칙기반 AI : rule-based system , 수동으로 규칙(특징)을 입력하면 기계가 추론, Prolog, LISP
##### [2] 학습기반 AI : 머신 러닝(Machine Learning), 데이터를 입력하면 기계가 스스로 새로운 특징을 학습하고 예측

####
#### 머신러닝 분류
##### [1] 지도학습(Supervised Learning) : 답이 주어진 상태에서 학습
- 회귀(Regression)
- 분류(Classification)
##### [2] 비지도학습(Unsupervised Learning) : 답을 모르고 학습
- 군집화(Clustering)
- 차원 축소(Dimension Reduction) : PCA(주성분 분석, Pricipal Component Analysis)
##### [3] 강화 학습(Reinforcement Learning) : 답을 모르고 있는 상태에서 답을 알아가는 강한 인공지능(자아를 갖음, 인간수준)
- 게임, 알파고(DQN)

####
#### 회귀(Reression) 모델
##### [1] 선형회귀(Linear Regrssion) : 1차 함수, 직선의 방정식
##### [2] 가중치(Weight) : 입력변수가 출력에 영향을 미치는 정도를 설정, 기울기 값
##### [3] 편향(Bias) : 기본 출력 값이 활성화되는 정도를 설정, y 절편
##### [4] 비용함수(Cost function) : 2차 함수, 포물선의 방정식
- cost(비용) = 오차 = 에러 = 손실(loss)
- cost(W,b) = (H(x) - y)^2
##### [5] 예측(가설,Hypothesis) 함수: predict
- H(x) = WX + b
##### [6] 경사하강법(Gradient Descent algorithm)
- 비용이 가장 작은 기울기 값을 구하는 알고리즘

