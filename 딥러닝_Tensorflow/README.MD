##  딥러닝(Tensorflow)
<br>
<br>

### \* 쥬피터 노트북에 tensorflow 1.15 설치 방법

##### python =3.7.x  
##### tensoflow = 1.15
<br>

#### [1] Anaconda 파이썬 버전 변경하기(파이썬 버전이 3.7.x 아래이거나 버전이 낮아 설치 안되는경우만 실행)
#### (파이썬 버전이 3.7.x 이상인 분은 [2] 번 부터 시작하세요)
<br>

#### * Anaconda Prompt를 관리자 권한으로 실행하고
#### conda create -n py376 python=3.7.6 anaconda    # 몇 분 소요됨  
#### conda activate py376
<br>

##### * 파이썬 버전 확인 명령
##### python -V
<br>

##### * 만일 원래 파이썬 버전으로 다시 사용하고 싶을 때는 deactivate를 실행한다
##### conda deactivate py376
<br>

#### [2] Anaconda Prompt를 관리자 권한으로 실행하고 아래 명령을 실행한다

#### python -m pip install --upgrade pip
#### pip3 install tensorflow==1.15
<br>

#### [3] 쥬피터 노트북 셀에서 에서 아래 코드를 실행시켜 버전을 확인해본다

##### * 파이썬 버전을 알아본다
#### import sys
##### print(sys.version)
<br>

##### * 텐서플로 버전을 알아본다
#### import tensorflow as tf
#### tf.\_\_version\_\_   
##### (주의: 언더바가 두개임) 
#### '1.15.0' 이 출력되면 설치 성공!!
<br>

### * AVX 미지원으로 인한 "DLL 초기화 루틴을 실행할 수 없습니다." 에러 해결방법 <br>
https://datamod.tistory.com/139
<br>

##### * 사용하고 있는 CPU 기술 탭의 Intel(R) Advanced Vector Extensions지원 여부는
##### pidkor47.msi(강사 깃허브에서)
##### 혹은 https://downloadcenter.intel.com/ko/download/28539?v=t 에서 다운로드 설치하여 확인힐 수 있다
<br>
  
#### * 파이썬 버전을 3.6.8로 추가 설치한다
#### * Anaconda Prompt를 관리자 권한으로 실행하고
#### conda create -n py368 python=3.6.8 anaconda    # 몇 분 소요됨  
#### conda activate py368
<br>

##### * 파이썬 버전 확인 명령
##### python -V
<br>

#### 텐서플로를 AVX를 지원하지 않는 1.6 보다 낮은 버전(1.5.0)으로 설치 한다
##### python -m pip install --upgrade pip
##### pip3 install tensorflow==1.5.0

<br>
