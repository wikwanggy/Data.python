import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path="python-data-analysis-master/data/drinks.csv"
drinks=pd.read_csv(file_path) 
print(drinks.info())

drinks.head(10)
drinks.describe()

# 두 피처 간의 상관 계수 구하기
# beer_servings, wine_servings 두 피처 간의 상관 계수를 계산합니다.
# pearson은 상관 계수를 구하는 계산 방법 중 하나를 의미하며, 가장 널리 쓰이는 방법
# corr=drinks[['beer_servings','wine_servings']].corr(method="pearson")
# print(corr)

# 여러 피처의 상관 관계 분석하기
#
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
# drinks['continent'] = drinks['continent'].fillna('OT')
# print(drinks.head(10))


# 파이차트로 시각화 하기
# labels = drinks['continent'].value_counts().index.tolist() #['AF','EU','AS','OT','OC','SA']
# fracs1 = drinks['continent'].value_counts().values.tolist() #[53,  45,  44,  26,  16,  12]
# explode = (0, 0, 0, 0.25, 0, 0)

# plt.pie(fracs1, explode=explode, labels=labels, autopct='%.0f%%', shadow=True)
# plt.title('null data to \'OT\'')
# plt.show()


# 대륙별 spriti_servings의 통계적 정보는 어느 정도 일까?
result=drinks.groupby("continent").spirit_servings.agg(["mean","min","max","sum"])
print(result.head())
# 전체 평균보다 많은 알코올을 섭취하는 나라는 어디일까?
# 총 알코올 소비량(total_liters_of_pure_alcoho)의 평균(전체)
total_mean = drinks.total_litres_of_pure_alcohol.mean()
#대륙별 총 알코올 소비량의 평균
continent_mean = drinks.groupby('continent')['total_litres_of_pure_alcohol'].mean()
#                                       대륙별      >= 전체평균
continent_over_mean = continent_mean[continent_mean >= total_mean]
print(continent_over_mean)
# 평균 beer_servings가 가장 높은 대륙은 어디일까?
# 대륙별                        맥주소비량의평균
beer_continent = drinks.groupby('continent').beer_servings.mean().idxmax()
print(beer_continent)


# 대륙별 spirit_servings의 평균, 최소, 최대, 합계를 시각화합니다.
n_groups = len(result.index)
means = result['mean'].tolist()
mins = result['min'].tolist()
maxs = result['max'].tolist()
sums = result['sum'].tolist()
 
index = np.arange(n_groups)
bar_width = 0.1
 
rects1 = plt.bar(index, means, bar_width,
                 color='r',
                 label='Mean')
 
rects2 = plt.bar(index + bar_width, mins, bar_width,
                 color='g',
                 label='Min')

rects3 = plt.bar(index + bar_width * 2, maxs, bar_width,
                 color='b',
                 label='Max')
 
rects3 = plt.bar(index + bar_width * 3, sums, bar_width,
                 color='y',
                 label='Sum')

plt.xticks(index, result.index.tolist())
plt.legend()
plt.show()