# titanic실습과제.py

# https://kaggle-kr.tistory.com/17

import pandas as pd
import seaborn as sb
titanic = sb.load_dataset('titanic')


# 1.1
titanic.to_csv('titanic_new.csv',index=False)
print(titanic.mean())

# 1.2
titanic_new = titanic.fillna(titanic.mean())
titanic_new.to_csv('titanic_new_no_nan.csv',index=False)

# 1.3

print('survived:',titanic['survived'][titanic['survived']
                                      == 1].count())
print('not survived:',titanic['survived'][titanic['survived']
                                          ==0].count())
# 1.4
print(titanic.pivot_table("survived", "class",aggfunc='mean'))

# 1.5
print(titanic.pivot_table("survived", "sex",aggfunc='mean'))

# 1.6
print(titanic.pivot_table("survived", "age",aggfunc='mean'))

# 1.7
print(titanic.pivot_table("survived", "sibsp",aggfunc='mean'))

# 1.8
print(titanic.pivot_table("fare","pclass",aggfunc='mean'))

# 1.9
new_df = titanic.drop(['deck'], axis = 1)
print(new_df)

# 1.10
print('-'*50)   # age를  3등급으로 나누어 등급의 성별 생존률
titanic["age_class"] = pd.qcut(titanic.age, 3,
                              labels = ["young", "midlle", "old"])
print(titanic.pivot_table('survived', index=['sex',
            'age_class'], aggfunc='mean', columns='pclass'))
print('-'*50)

