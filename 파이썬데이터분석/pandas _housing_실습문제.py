# pandas _housing_실습문제.py
import numpy as np
import pandas as pd

# 1번
df = pd.read_csv('boston_train.csv')

print("Shape:\n", df.shape)
print("Length:\n", len(df))
print("Column Headers:\n", df.columns)
print("Data types:\n", df.dtypes)
print("Index:\n", df.index)
print("Values:\n", df.values)
print("head",df.head())
print("teail",df.tail())
print("Describe", df.describe(),"\n")
print("Non NaN observations", df.count(),"\n")
print("MAD", df.mad(),"\n")
print("Median", df.median(),"\n")
print("Mean", df.mean(),"\n")
print("Sum", df.sum(),"\n")
print("Min", df.min(),"\n")
print("Max", df.max(),"\n")
print("Mode", df.mode(),"\n")
print("Standard Deviation", df.std(),"\n")
print("Variance", df.var(),"\n")
print("Skewness", df.skew(),"\n")
print("Kurtosis", df.kurt(),"\n")

print('-'*50)


# 2-1번
print('CRIM mean: ', df['CRIM'].mean())
print(df[df['CRIM'] > df['CRIM'].mean()])

print('-'*50)

# 2-2번
print('AGE mean: ', df['AGE'].mean())
print(df[df['AGE'] < df['AGE'].mean()])

print('-'*50)

# 2-3번
print('MEDV median: ', df['MEDV'].median())
print(df[df['MEDV'] < df['MEDV' ].median()])

print('-'*50)

# 3번

df_train = pd.read_csv('boston_train.csv')
df_test = pd.read_csv('boston_test.csv')

# df = pd.concat([df_train[:10],df_test[:10]],ignore_index=True)
df = df_train[:10].append(df_test[:10],ignore_index=True)

print(df)

df.to_csv('boston_batch.csv', index=False)
print('-'*50)

# 4번

df = pd.read_csv('boston_train.csv')

# print(pd.pivot_table(df,columns=['CRIM'], agg_func = np.sum))

for col in df.columns:
# for col in df.columns[:]:
    print('[',col,']')
    print('sum :',df[col].sum())
    print('mean :',df[col].mean())
    print('median :',df[col].median())
    print('min :',df[col].min())
    print('max :',df[col].max())
    print('describe :', df[col].describe())

# 5번

sunspots = pd.read_csv("sunspots.csv")
print("Total Null Values\n", pd.isnull(sunspots).sum())
sunspots = sunspots.fillna(0)
print(sunspots)
print("Total Null Values\n", pd.isnull(sunspots).sum())

sunspots.to_csv('sunspots_new.csv',float_format = '%.2f',
                na_rep='NaN',index=False)

# 6번
sunspots = pd.read_csv("sunspots.csv")
sunspots['Date'] = pd.to_datetime(sunspots['Date'])
print(sunspots.dtypes)
print('Mean:',pd.Series(sunspots['Date']).mean())

sunspots = sunspots[sunspots['Date'] > sunspots['Date'].mean()]

sunspots.to_csv('sunspots_new2.csv',float_format = '%.2f',
                na_rep='NaN',index=False)
sunspots = pd.read_csv("sunspots_new2.csv")
print(sunspots.dtypes)

print('-'*50)

# 7번
df_A_B = pd.DataFrame({'판매월': ['1월', '2월', '3월', '4월'],
                       '제품A': [100, 150, 200, 130],
                       '제품B': [90, 110, 140, 170]})
print(df_A_B)

df_C_D = pd.DataFrame({'판매월': ['1월', '2월', '3월', '4월'],
                       '제품C': [112, 141, 203, 134],
                       '제품D': [90, 110, 140, 170]})
print(df_C_D)

print(df_A_B.merge(df_C_D,on='판매월'))

print('-'*50)

# 8번
df_left = pd.DataFrame({'key':['A','B','C'],
                        'left': [1, 2, 3]})
print(df_left)

df_right = pd.DataFrame({'key':['A','B','D'],
                         'right': [4, 5, 6]})
print(df_right)

# 8.1
print('\ninner join:')
print(df_left.merge(df_right, how='inner', on = 'key'))

# 8.2
print('\nouter join:')
print(df_left.merge(df_right, how='outer', on = 'key'))

# 8.3
print('\nleft join:')
print(df_left.merge(df_right, how='left', on = 'key'))

# 8.4
print('\nright join:')
print(df_left.merge(df_right, how='right', on = 'key'))


print('-'*50)


# 9번
df = pd.read_csv('WHO_first9cols.csv')
result = df['Country'].str.contains("Albania")
print(df['Country'][result])  # 요소값들로 출력
print(df[result])  # 데이터 프레임으로 출력

result = df['Country'].str.contains("Ethiopia")
print(df['Country'][result])  # 요소값들로 출력
print(df[result]) # 데이터 프레임으로 출력







