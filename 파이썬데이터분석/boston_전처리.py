# boston_전처리.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('boston_train.csv')
# df = df.fillna(0)

# 'CRIM' 원본
crime = df['CRIM'].values
plt.subplot(221)
plt.hist(crime)
plt.title('CRIME 원본')
# plt.show()

# 'CRIM' 표준화
crime = crime.reshape(-1,1)
crime_std = StandardScaler().fit_transform(crime)
crime_std_zoomin = crime_std[crime_std < 2]

plt.subplot(222)
plt.hist(crime_std_zoomin)
plt.title('CRIME 표준화')


# 'CRIM' 정규화
crime_std_zoomin = crime_std_zoomin.reshape(-1,1)
crime_minmax = MinMaxScaler().fit_transform(crime_std_zoomin)

plt.subplot(224)
plt.hist(crime_minmax)
plt.title('CRIME 정규화')

plt.show()


# 'ZN' 원본
zn = df['ZN'].values
plt.subplot(221)
plt.hist(zn)
plt.title('zn 원본')

# 'ZN' 표준화
zn = zn.reshape(-1,1)
zn_std = StandardScaler().fit_transform(zn)
# plt.boxplot(zn_std)
# plt.show()
# input()
zn_std_zoomin = zn_std[zn_std < 1]

plt.subplot(222)
plt.hist(zn_std_zoomin)
plt.title('ZN 표준화')


# 'ZN' 정규화
zn_std_zoomin = zn_std_zoomin.reshape(-1,1)
zn_minmax = MinMaxScaler().fit_transform(zn_std_zoomin)

plt.subplot(224)
plt.hist(zn_minmax)
plt.title('ZN 정규화')

plt.show()

# 'TAX' 원본
tax = df['TAX'].values
plt.subplot(221)
plt.hist(tax)
plt.title('TAX 원본')

# 'TAX' 표준화
tax = tax.reshape(-1,1)
tax_std = StandardScaler().fit_transform(tax)
tax_std_zoomin = tax_std[tax_std < 0.5]

plt.subplot(222)
plt.hist(tax_std_zoomin)
plt.title('TAX 표준화')


# 'TAX' 정규화
tax_std_zoomin = tax_std_zoomin.reshape(-1,1)
tax_minmax = MinMaxScaler().fit_transform(tax_std_zoomin)

plt.subplot(224)
plt.hist(tax_minmax)
plt.title('TAX 정규화')
plt.show()

# 'MEDV' 원본
medv = df['MEDV'].values
plt.subplot(221)
plt.hist(medv)
plt.title('MEDV 원본')

# 'MEDV' 표준화
medv = medv.reshape(-1,1)
medv_std = StandardScaler().fit_transform(medv)
medv_std_zoomin = medv_std[medv_std < 1.8]

plt.subplot(222)
plt.hist(medv_std_zoomin)
plt.title('MEDV 표준화')


# 'MEDV' 정규화
medv_std_zoomin = medv_std_zoomin.reshape(-1,1)
medv_minmax = MinMaxScaler().fit_transform(medv_std_zoomin)

plt.subplot(224)
plt.hist(medv_minmax)
plt.title('MEDV 정규화')
plt.show()


# 최종 결과
plt.subplot(221)
plt.hist(crime_minmax)
plt.title('CRIM 정규화')

plt.subplot(222)
plt.hist(zn_minmax)
plt.title('ZN 정규화')

plt.subplot(223)
plt.hist(tax_minmax)
plt.title('TAX 정규화')

plt.subplot(224)
plt.hist(medv_minmax)
plt.title('MEDV 정규화')
plt.show()
