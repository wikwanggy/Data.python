import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path="python-data-analysis-master/data/drinks.csv"
drinks=pd.read_csv(file_path) 
# print(drinks.info())

# drinks.head(10)
# drinks.describe()

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


# # 대륙별 spriti_servings의 통계적 정보는 어느 정도 일까?
# result=drinks.groupby("continent").spirit_servings.agg(["mean","min","max","sum"])
# # print(result.head())
# # 전체 평균보다 많은 알코올을 섭취하는 나라는 어디일까?
# # 총 알코올 소비량(total_liters_of_pure_alcoho)의 평균(전체)
# total_mean = drinks.total_litres_of_pure_alcohol.mean()
# #대륙별 총 알코올 소비량의 평균
# continent_mean = drinks.groupby('continent')['total_litres_of_pure_alcohol'].mean()
# #                                       대륙별      >= 전체평균
# continent_over_mean = continent_mean[continent_mean >= total_mean]
# # print(continent_over_mean)
# # 평균 beer_servings가 가장 높은 대륙은 어디일까?
# # 대륙별                        맥주소비량의평균
# # beer_continent = drinks.groupby('continent').beer_servings.mean().idxmax()
# # print(beer_continent)


# # 대륙별 spirit_servings의 평균, 최소, 최대, 합계를 시각화합니다(차트).
# n_groups = len(result.index) # n_groups는 위에 선언한 result의 index값의 개수인 6개이다.
# means = result['mean'].tolist() # 평균의 값들을 tolist로 강제 변환하여 저장한것을 means에 저장
# mins = result['min'].tolist() # 최소값의 값들을 tolist로 강제 변환하여 저장한것을 mins에 저장
# maxs = result['max'].tolist() # 최대값의 값들을 tolist로 강제 변환하여 저장한것을 maxs에 저장
# sums = result['sum'].tolist() # 합계의 값들을 tolist로 강제 변환하여 저장한것을 sums에 저장
 
# index = np.arange(n_groups)# np.arange(n_groups) index에 저장해라
# bar_width = 0.1 # bar width 굵기
#  # plt.bar(세로막대), index,means,bar_width를 호출
# rects1 = plt.bar(index, means, bar_width, 
#                  color='r', # 색상은 레드,
#                  label='Mean') # label은 Mean(평균)
#  # plt.bar(세로막대), index,min,bar_width를 호출
# rects2 = plt.bar(index + bar_width, mins, bar_width,  # index가 겹치지 않게 index+bar_width
#                  color='g', # 색상은 그린
#                  label='Min') # 라벨은 min(최소)
#  # plt.bar(세로막대), index,max,bar_width를 호출
# rects3 = plt.bar(index + bar_width * 2, maxs, bar_width,# index가 겹치지 않게 index+bar_width *2
#                  color='b', # 색상은 블루
#                  label='Max') # 라벨은 max(최대)
#  # plt.bar(세로막대), index,sum,bar_width를 호출
# rects3 = plt.bar(index + bar_width * 3, sums, bar_width,  # index가 겹치지 않게 index+bar_width *3
#                  color='y', # 색상은 옐로우 
#                  label='Sum') #라벨은 sum(합계)

# plt.xticks(index, result.index.tolist()) # index와 result.index.tolist()
# plt.legend() # 범례 표시하기
# plt.show() # 차트를 보여주게 하여라

# # 대륙별 total_litres_of_pure_alcohol을 시각화합니다.
# # 대륙병 평균값을 tolist로 변환 후 continents에 저장
# continents = continent_mean.index.tolist()
# # 차트에서 빨간색 부분(전체 평균 추가(append))
# continents.append('mean')
# #x_pos의 시작점과 긑을 continents의 길이로 하라(6)
# x_pos = np.arange(len(continents))
# #  대륙 평균값을 tolist로 변환 후 alcohol에 저장
# alcohol = continent_mean.tolist()
# # 전체 평균 추가(append(total_mean))
# alcohol.append(total_mean)
# #                   길이 대륙별 평균값,전체평균값      가운데            투명도
# bar_list = plt.bar(x_pos,       alcohol,        align='center',   alpha=0.5)
# # continent에 append를 해서 6->7로 늘어남 그렇기에 -1을 해준다.
# bar_list[len(continents) - 1].set_color('r')
# # 표에 가운데에 있는 "------" "k--"이다 k는 검정을 나타내고 --은 선을 나타낸다.
# plt.plot([0., 6], [total_mean, total_mean], "k--")
# # x축엔 continents의 index와 추가한 mean을 표시하고 continents의 값들을 표시
# plt.xticks(x_pos, continents) 
# # y축에 라벨 추가
# plt.ylabel('total_litres_of_pure_alcohol') 
#  # 제목 추가
# plt.title('total_litres_of_pure_alcohol by Continent')
# # 차트를 보여주기
# plt.show() 

# # 대륙별 beer_servings을 시각화합니다.
# #
# beer_group = drinks.groupby('continent')['beer_servings'].sum()
# #
# continents = beer_group.index.tolist()
# #y_pos의 시작점과 끝은 continents의 길이(6)이다.
# y_pos = np.arange(len(continents))

# alcohol = beer_group.tolist()
# #                   길이  대륙별 맥주소비량,전체 소비량      가운데            투명도
# bar_list = plt.bar(y_pos,        alcohol,             align='center',      alpha=0.5)
# # "EU"를 가진 인덱스는 색상을 red로 표시
# bar_list[continents.index("EU")].set_color('r')
# # y_pos의 시작점과 끝은 continents의 길이(6)로 하고 continets의 값들을 표시
# plt.xticks(y_pos, continents)
# # y축 라벨 추가
# plt.ylabel('beer_servings')
# # 제목 추가
# plt.title('beer_servings by Continent') 
# # 차트를 보여주기
# plt.show()


# 아프리카와 유럽간의 맥주 소비량 차이를 검정합니다.
# 아프리카
africa = drinks.loc[drinks['continent']=='AF']
# 유럽
europe = drinks.loc[drinks['continent']=='EU']

# print(drinks["continent"]=="AF")
# print(africa)
# print(europe)

from scipy import stats
# tTestResult = stats.ttest_ind(africa['beer_servings'], europe['beer_servings'])
# tTestResultDiffVar = stats.ttest_ind(africa['beer_servings'], europe['beer_servings'], equal_var=False)

# print("The t-statistic and p-value assuming equal variances is %.3f and %.3f." % tTestResult)
# print("The t-statistic and p-value not assuming equal variances is %.3f and %.3f" % tTestResultDiffVar)

# total_servings 피처를 생성합니다.
#총 소비량(total_servings)는=맥주 소비량(bee                 r_servings)+와인 소비량(wine_servings)+spirit소비량(spirit_servings)
drinks['total_servings'] = drinks['beer_servings'] + drinks['wine_servings'] + drinks['spirit_servings']

# 술 소비량 대비 알콜 비율 피처를 생성합니다.
# 술 소비량 대비 알콜 비율 피처=총 알콜 비율(total_litres_of_pure_alcohol)/총 소비량(total_servings)
drinks['alcohol_rate'] = drinks['total_litres_of_pure_alcohol'] / drinks['total_servings']
# 술 소비량 대비 알콜 비율 피처에['alcohol_rate']에 null값이 있으면 0으로 채워라(fillna())
drinks['alcohol_rate'] = drinks['alcohol_rate'].fillna(0)

# 순위 정보를 생성합니다.
country_with_rank = drinks[['country', 'alcohol_rate']]
# sort_values(내림차순)(by=[기준'alcohol_rate'], ascending=0)
country_with_rank = country_with_rank.sort_values(by=['alcohol_rate'], ascending=0)
# 상위 5위까지만 보여주기
print(country_with_rank.head(5))

# 국가별 순위 정보를 그래프로 시각화합니다.
# country_with_rank.country.tolist()를 contry_list에 저장
country_list = country_with_rank.country.tolist()
# x_pos의 길이는 contry_list의 index의 길이로 한다
x_pos = np.arange(len(country_list))
# country_with_rank.alcohol_rate.tolist()를 rank에 저장
rank = country_with_rank.alcohol_rate.tolist()
# bar에 x포지션, x축엔 rank의 인덱스를 표시
bar_list = plt.bar(x_pos, rank)
# bar_list에 있는 index 중에 South Korea는 red로 표시
bar_list[country_list.index("South Korea")].set_color('r')
# y라벨은 술 소비량 대비 알콜 비율로 표시한다
plt.ylabel('alcohol rate')
# 제목 추가
plt.title('liquor drink rank by contry')
#axis() - X, Y축이 표시되는 범위를 지정하거나 반환합니다.
plt.axis([0, 200, 0, 0.3])
# contry_list에 index중에 "South Korea"를 korea_rank에 저장
korea_rank = country_list.index("South Korea")

korea_alc_rate = country_with_rank[country_with_rank['country'] == 'South Korea']['alcohol_rate'].values[0]
# annotate 함수는 주석달기
#               텍스트    : + str으로 변환한 korea_rank에 +1
plt.annotate('South Korea : ' + str(korea_rank + 1), 
            # 화살이 가르칠 위치
             xy=(korea_rank, korea_alc_rate), 
            # xytext=주석을 표시할 xy좌표를 설정할 때 사용합니다.
             xytext=(korea_rank + 10, korea_alc_rate + 0.05),
            # 화살표 추가(색상은 red 텍스트로부터 0.05 떨어진 위치)
             arrowprops=dict(facecolor='red', shrink=0.05))
# 차트 보여주기
plt.show() 