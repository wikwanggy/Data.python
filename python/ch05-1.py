import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 중고나라 데이터셋 살펴보기
df= pd.read_csv("python-data-analysis-master/data/used_mobile_phone.csv")
print(df.info())
print(df.head())

# 개별 피처 탐색하기 : date 피처 탐색
# create_date로 부터 '월'을 의미하는 month 정보를 치퍼로 추출합니다(람다).
df['month']=df['create_date'].apply(lambda x: x[:7]) 

# 월별 거래 횟수를 계산하여 출력합니다.
print(df['month'].value_counts())

# 개별 피처 탐색하기 : date 피처 탐색 
# 일별 거래 횟수를 계산하여 그래프로 출력합니다.
df_day = pd.to_datetime(df['create_date'].apply(lambda x: x[:10])).value_counts()
df_day.plot()
plt.show()

# 개별 피처 탐색하기 : price 피처 탐색
# 가격의 분포를 그래프로 탐색합니다.
df['price'].hist(bins="auto")
plt.show()

# 개별 피처 탐색하기 : price 피처 탐색
 # 휴대폰 기종(phone_model)별 가격의 평균과 표준편차를 계산합니다.
df_price_model_mean = df.groupby('phone_model')['price'].transform(lambda x: np.mean(x))
df_price_model_std = df.groupby('phone_model')['price'].transform(lambda x: np.std(x))

# 이를 바탕으로 모든 데이터의 z-score를 계산합니다. 이는 해당 데이터의 가격이 기종별 평균에
# 비해 어느 정도로 높거나 늦은지를 알 수 있게 하는 점수입니다.
df_price_model_z_score=(df['price'] - df_price_model_mean) / df_price_model_std
df_price_model_z_score.hist(bins='auto')
plt.show()

# 개별 피처 탐색하기 : factory_price 피처 탐색
# factory_price 피처의 분포를 탐색합니다.
df['factory_price'].hist(bins='auto')
plt.show()

# factory_price 와 price 피처를 산점도 그래츠로 출력하여 상관 관게를 살펴봅니다.
df.plot.scatter(x='factory_price',y='price')
plt.show()

# 개별 피처 탐색하기 : phone_model 피처 탐색
# 기종별 총 거래 데이터 개수를 집계합니다.
model_counts = df['phone_model'].value_counts()
print(model_counts.describe())

# 기종별 총 거래 데이터 개수를 상자 그림으로 살펴봅니다.
plt.boxplot(model_counts)

# Randmo forest regressor를 이용한 가격예측
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# 데이터를 학습/테스트용 데이터로 분리합니다 
df = df[['price', 'phone_model', 'factory_price', 'maker', 'price_index', 'month']]
df = pd.get_dummies(df, columns=['phone_model', 'maker', 'month'])
X = df.loc[:, df.columns != 'price']
y= df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 랜덤 포레스트 모델을 학습합니다.
forest = RandomForestRegressor(n_estimators=1000, criterion='mse')
forest.fit(X_train,y_train)
y_train_pred =forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# 학습한 모델을 평가합니다.
print('MSE train : %.3f, test: %.3f'%(mean_squared_error(y_train, y_train_pred),
                                      mean_squared_error(y_test, y_test_pred)))
print('R^2 train : %.3f, test : %.3f' %(r2_score(y_train,y_train_pred),
                                        r2_score(y_test, y_test_pred)))

# 피처의 중요도 분석하기
# 학습한 모델의 피처 중요도를 그래프로 살펴봅니다.
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices])
plt.show()

#학습한 모델의 피처 중요도를 출력합니다.
feat_labels = X.columns.tolist()
feature = list(zip(feat_labels, forest.feature_importances_))
sorted(feature, key=lambda tup: tup[1], reverse=True)[:10]

# 피처 중요도 분석하기 
# month 피처 중, 영향력이 높은순으로 정렬하여 출력합니다.
for sorted_feature in sorted(feature, key=lambda tup: tup[1], reverse=True):
    if "month" in sorted_feature[0]:
        print(sorted_feature)
        

#Step 2 피처 엔지니어링 : 예측 모델 개선하기
# 기존 피처 가공하기 : 'create_date'

#데이터를 다시 불러옵니다.
df = pd.read_csv("python-data-analysis-master/data/used_mobile_phone.csv")

from datetime import datetime
import time
# create_date 피처를 수치적으로 계산하기 위해 unixtime으로 변환하는
# 함수를 정의합니다.
def date_to_unixtime(date_str):
    timestamp = time.mktime(datetime.strptime(date_str, '%Y-%m-%d').timetuple())
    return timestamp
# create_date 피처를 '현재와 얼마나 가까운 데이터인지; 판단하기 위한
# 점수를 생성합니다. 먼저 unixtime으로 데이터를 변환합니다
df['create_unixtime']= df['create_date'].apply(lambda x: date_to_unixtime(x[:10]))

# 변환된 unixtime에 min-max 스케일링을 적용합니다
df['create_time_score']= (df['create_unixtime']-df['create_unixtime'].min()) / (df['create_unixtime'].max() - df['create_unixtime'].min())
print(df[['create_date','create_unixtime', 'create_time_score']].head())

# 기존 피처의 가공 : phone_model
# phone_model 피처에서 저장 용량(phone_model_storage) 피처를 추출합니다
df['phone_model_storage'] = df['phone_model'].apply(lambda x: x.split(" ")[1])

# phone_model 피처에서 기종 세부명(phone_model_detail) 피처를 추출합니다.
df['phone_model_detail'] = df['phone_model'].apply(lambda x: ' '.join(x.split(" ")[:-1]))
print(df[['phone_model_storage','phone_model_detail']].head())

# 기존 피처의 가공 : phone_model
# phone_model 피처의 기종별 거래 데이터 개수를 집계합니다.
model_counts = df['phone_model'].value_counts()

# phone_model_detail 피처의 기종별 거래 데이터 개수를 집계합니다.
model_detail_counts = df['phone_model_detail'].value_counts()
data = [model_counts, model_detail_counts]

# 두 피처 간의 기종별 거래 데이터 개수를 비교합니다
mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
ax.boxplot(data)
