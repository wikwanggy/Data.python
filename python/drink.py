import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path="python-data-analysis-master/data/drinks.csv"
drinks=pd.read_csv(file_path) 
print(drinks.info())

# drinks.head(10)
# drinks.describe()

# # 두 피처 간의 상관 계수 구하기
# # beer_servings, wine_servings 두 피처 간의 상관 계수를 계산합니다.
# # pearson은 상관 계수를 구하는 계산 방법 중 하나를 의미하며, 가장 널리 쓰이는 방법
# corr=drinks[['beer_servings','wine_servings']].corr(method="pearson")
# print(corr)

# # 여러 피처의 상관 관계 분석하기
# #
# cols=['beer_servings','spirit_servings','wine_servings','total_litres_of_pure_alcohol']
# corr=drinks[cols].corr(method="pearson")
# print(corr)

import seaborn as sns
import matplotlib.pyplot as plt

# corr 행렬 히트맵을 시각화 합니다.

# 그래프 출력을 위한 cols 이름을 축약합니다.
# cols_view=['beer','spirit','wine','alcohol'] 
# sns.set(font_scale=1.5)
# hm = sns.heatmap(corr.values,
#     cbar= True, # 사이드바(cbar)
#     annot=True, # 네모안에 숫자 표시 
#     square=True, # 정사각형으로 출력
#     fmt='.2f',
#     annot_kws={'size':15},
#     yticklabels=cols_view,
#     xticklabels=cols_view)

# plt.tight_layout() 
# plt.show()

# 시각화 라이브러리를 이용한 피처간의 산점도 그래프를 출력합니다
# sns.set(style='whitegrid',context='notebook')
# sns.pairplot(drinks[['beer_servings','spirit_servings','wine_servings','total_litres_of_pure_alcohol']],height=2.5)

# plt.show()
# null값인지 있는 값인지 확인하기.
# print(drinks['continent'].head(10).isnull())
# 결측데이터를 처리합니다 : 정보가 없는 국가를 'Others', 줄여서 통합 -> 'OT'
drinks['continent'] = drinks['continent'].fillna('OT')
print(drinks.head(10))


# 파이차트로 시각화 하기
labels = drinks['continent'].value_counts().index.tolist() #['AF','EU','AS','OT','OC','SA']
fracs1 = drinks['continent'].value_counts().values.tolist() #[53,  45,  44,  26,  16,  12]
explode = (0, 0, 0, 0.25, 0, 0)

plt.pie(fracs1, explode=explode, labels=labels, autopct='%.0f%%', shadow=True)
plt.title('null data to \'OT\'')
plt.show()