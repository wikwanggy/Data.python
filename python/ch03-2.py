import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path="python-data-analysis-master/data/market-price.csv"

bitcoin_df=pd.read_csv(file_path, names=['day','price'])

# 기본 정보를 출력합니다.
print(bitcoin_df.shape)
print(bitcoin_df.info())
print(bitcoin_df.tail())

# to_datetime으로 day 피처를 시계열 피처로 변환합니다. 
# bitcoin_df['day'] = pd.to_datetime(bitcoin_df['day'])

# day 데이터 프레임의 index로 설정합니다.
bitcoin_df.index=bitcoin_df["day"]
bitcoin_df.set_index("day",inplace=True)
print(bitcoin_df.head())
print(bitcoin_df.describe())

# 일자별 비트코인 시세를 시각화 합니다.
# bitcoin_df.plot()
# plt.show()

# ARIMA 모델 활용하기위한 모듈 추가
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

# (AR=2, 차분=1, MA=2)파라미터로 ARIMA 모델을 학습한다.
model = ARIMA(bitcoin_df.price.values, order=(2,1,2))
model_fit = model.fit(trend='c', full_output=True, disp=True)
print(model_fit.summary())
