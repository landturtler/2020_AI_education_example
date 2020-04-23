# Series_실습문제.py


import pandas as pd

def series_lab01():
    sr = pd.Series([0], name='제품 데이터',
                   index = ['제품명'] )
    while True:
        print('제품수량관리')
        print('=' * 60)
        print('1. 입력 2. 출력 3.검색  9.종료 중에 메뉴를 선택하세요')
        print('=' * 60)

        menu = input('입력하세요  ->   ')
        menu = int(menu)

        if menu == 1:
            while True:
                name = input('제품명:')
                count = input('수량:')
                count = int(count)
                sr = sr.append(pd.Series([count], index=[name]))
                contin = input('계속 입력? Y/N')
                if contin == 'N' or contin == 'n':
                    break

        elif menu == 2:
            print('=' * 30)
            print('      제품명   수량')
            print('=' * 30)
            for idx, v in sr.items():
                print('%10s%10d' % (idx, v))

        elif menu == 3:
            name = input('제품명:')
            print('=' * 30)
            print('      제품명   수량')
            print('=' * 30)

            print('\t',sr[sr.index.str.contains(name)])

        elif menu == 9:
            print('종료합니다')
            break;

        else:
            pass


# series_lab01()

# 2번
# 방법 1
def change_nan_by_mean_df():
    org_csv = 'WHO_first9cols.csv'
    new_csv = 'WHO_NoNaN.csv'
    df = pd.read_csv(org_csv)

    sr_null = pd.isnull(df).sum() # 결측치 수를 얻어서 출력
    print('[',org_csv,'결측치수]')
    print(sr_null)

    df = df.fillna(df.mean()) # 결측치를 평균값으로 변경

    df.to_csv(new_csv,index=False) # 새파일로 저장

    df = pd.read_csv(new_csv)
    sr_null = pd.isnull(df).sum() # 결측치 수를 얻어서 출력
    print('[',new_csv,'결측치수]')
    print(sr_null)


change_nan_by_mean_df()

# 방법 2
def change_nan_by_mean_for_loop():
    org_csv = 'WHO_first9cols.csv'
    new_csv = 'WHO_NoNaN.csv'
    df = pd.read_csv(org_csv)
    sr = pd.isnull(df).sum()
    print('[',org_csv,'결측치수]')
    print(sr)  # 결측치 수를 출력

    for n in range(len(df.columns)) :
        if(sr[n]>0 and df[sr.index[n]].dtype != 'object' ):
            col_mean = df[sr.index[n]].mean()  # 컬럼의 평균값을 구한다
            # print(n,col_mean)
            df[sr.index[n]] = df[sr.index[n]].fillna(col_mean)

    df.to_csv(new_csv,index=False)

    df = pd.read_csv(new_csv)
    sr = pd.isnull(df).sum()
    print('[',new_csv,'결측치수]')
    print(sr)  # 결측치 수를 출력

change_nan_by_mean_for_loop()
