##  딥러닝(Tensorflow)
<br>
<br>
-------------------------------------------------------------------------------------------------<br>

### 딥러닝 강좌
#### 모두를 위한 머신러닝과 딥러닝의 강의(김성훈교수님)

https://hunkim.github.io/ml/

-------------------------------------------------------------------------------------------------<br>
<br>

# 윈도우에 PyCharm 설치 환경에 텐서플로 1.15.0 설치하기

#### [1] python 3.6.7 설치
https://www.python.org/downloads/release/python-376/
#### 사이트에서 하단 Files 목록에서 Windows x86-64 executable installer을 선택 클릭하여
#### 'python-3.7.6-amd64.exe' 파일이 다운로드된다 

#### 설치 시작 화면에서 맨 아래 'Add Python 3.7 to PATH' 를 체크해주고 'Install Now'를 선택하고 
#### 다른 옵션 선택없이 그대로 설치를 진행한다
#### 설치가 끝나면 윈도우 시작메뉴에 Pyhon 3.7 이 새로 설치되어 추가된걸 확인 할 수 있을 것이다

<br>

#### [2] PyCharm 다운로드 및 설치

https://www.jetbrains.com/ko-kr/pycharm/download/#section=windows
#### 에서 오른쪽의 'Community' 의 '다운로드' 버튼을 눌러 'pycharm-community-2020.1.exe' 파일을 다운받아 실행시킨다
#### 별다른 옵션 없이 그대로 설치해준다
<br>

#### [3] tensorflow 1.15.0 설치하기
#### 아래 주소를 참조하여 설치한다 tensorflow를 2.0 버전이 아닌 1.15.0으로 설치 하도록 한다
####  https://webnautes.tistory.com/1173
####  수업 중 강사 지시에 따라 함께 설치하도록 합니다
<br>

-------------------------------------------------------------------------------------------------<br>
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
#### 텐서플로 1.15.0 버전으로 설치
#### python -m pip install --upgrade pip
#### pip3 install tensorflow==1.15
<br>

#### [3] 쥬피터 노트북 셀에서 에서 아래 코드를 실행시켜 버전을 확인해본다

##### * 파이썬 버전을 알아본다
#### import sys
#### print(sys.version)

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

#### * 텐서플로를 AVX를 지원하지 않는 1.6 보다 낮은 버전(1.5.0)으로 설치 한다
##### python -m pip install --upgrade pip
##### pip3 install tensorflow==1.5.0  # 장시간 소요됨

<br>

### * 아나콘다 가상환경 제거
####  
#### conda env list
#### conda remove --name py368 --all
----------------------------------------------------------------------------------------------------------<br>

