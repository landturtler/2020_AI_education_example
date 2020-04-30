##  딥러닝(Tensorflow)
<br>

#### <이 문서는 현재 수정 중이며 아직 패키지의 버전 값 갱신이 필요합니다>

<br>

### \* 쥬피터 노트북에 tensorflow 1.15 설치 방법

##### python =3.7.x  
##### tensoflow = 1.15

<br>

#### [1] Anaconda 파이썬 버전 변경하기(파이썬 버전이 3.7.2)

Anaconda Prompt를 관리자 권한으로 실행하고
#### conda create -n py372 python=3.7.2 anaconda    # 몇 분 소용됨  
#### conda activate py372

##### 파이썬 버전 확인 명령
##### python -V
##### 만일 원래 파이썬 버전으로 다시 사용하고 싶을 때는 deactivate를 실행한다
##### conda deactivate py372
<br>

#### [2] Anaconda Prompt를 관리자 권한으로 실행하고 아래 명령을 실행한다

#### python -m pip install --upgrade pip
#### pip3 install tensorflow==1.15
<br>

#### [3] 쥬피터 노트북 셀에서 에서 아래 코드를 실행시켜 버전을 확인해본다

##### 파이썬 버전을 알아본다
#### import sys
##### print(sys.version)

##### 텐서플로 버전을 알아본다
#### import tensorflow as tf
#### tf.\_\_version\_\_   
##### (주의: 언더바가 두개임) 
#### '1.15.0' 이 출력되면 설치 성공!!
