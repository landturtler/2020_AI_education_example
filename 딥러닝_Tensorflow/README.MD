##  딥러닝(Tensorflow)
<br>

### \* 쥬피터 노트북에 tensorflow 1.15 설치 방법

##### python =3.7.4  (주의: 3.7.4 이후 버전에서는 설치안됨)
##### tensoflow = 1.15

#### [1] Anaconda Prompt를 관리자 권한으로 실행하고 아래 명령을 실행한다

#### python -m pip install --upgrade pip
#### pip3 install tensorflow==1.15

#### [2] 쥬피터 노트북 셀에서 에서 아래 코드를 실행시켜 버전을 확인해본다

#### import tensorflow as tf
#### tf.\_\_version\_\_   
##### (주의: 언더바가 두개임) 
#### '1.15.0' 이 출력되면 설치 성공!!
