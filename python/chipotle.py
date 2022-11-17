import pandas as pd


file_path="python-data-analysis-master/data/chipotle.tsv"
#read.csv() 함수로 데이터를 데이터 프레임 형태(엑셀)로 불러옵니다.
chipo=pd.read_csv(file_path, sep="\t")

print(chipo.shape)
print("--------------------------------")
print(chipo.info())

# chipo라는 데이터 프레임에서 순서대로 10개의 데이터를 보여줍니다.
chipo.head(10)

print(chipo.columns)
print("--------------------------------")
print(chipo.index)

#order_id는 숫자의 의미를 가지지 않기 때문에 str으로 변환합니다.
chipo['order_id']=chipo['order_id'].astype(str)
# chipo 데이터 프레임 에서 수치형 피처들의 기초 통께량을 확인합니다
print(chipo.describe()) 

print(len(chipo['order_id'].unique())) # order_id의 개수를 출력합니다.
print(len(chipo['item_name'].unique())) # item_name의 개수를 출력합니다.

# 가장 많이 주문한 아이템 Top 10 출력하기
item_count=chipo['item_name'].value_counts()[:10]
for idx, (val, cnt) in enumerate(item_count.iteritems(), 1):
    print("TOP", idx, ":", val,cnt)
    
#아이템별 주문 개수와 총량구하기
order_count=chipo.groupby('item_name')['order_id'].count()
order_count[:10] # 아이템별 주문 개수를 출력합니다
item_quantity=chipo.groupby('item_name')['quantity'].sum()
item_quantity[:10] #아이템별 주문 총량을 출력합니다

import numpy as np
import matplotlib.pyplot as plt

item_name_list = item_quantity.index.tolist()
x_pos=np.arange(len(item_name_list))
order_cnt=item_quantity.values.tolist()

plt.bar(x_pos, order_cnt, align="center")
plt.ylabel("ordered_item_count")
plt.title('Distribution of all orderd item')

plt.show()

# 데이터 전처리: 함수 사용하기
print(chipo)
print("-----------")
chipo["item_price"].head()

#colum 단위 데이터에 apply() 함수로 전처리를 적용합니다.
chipo['item_price']=chipo['item_price'].apply(lambda x: float(x[1:]))
print(chipo.describe())

# 주문당 평균 계산금액 출력하기
chipo.groupby('order_id')['item_price'].sum().mean()

# 한 주문에 10달러 이상 지불한 주문번호(id) 출력하기
chipo_orderid_group=chipo.groupby("order_id").sum()
results=chipo_orderid_group[chipo_orderid_group.item_price >= 10]
print(results[:10])
print(results.index.values)

# 각 아이템 가격
# 동일 아이템을 1개만 구매한 주문
chipo_one_item=chipo[chipo.quantity==1]
# 아이템이름별 최저가를 계산
price_per_item=chipo_one_item.groupby("item_name").min()
# 가격을 기준으로 정렬(ascending=False(내림차순) ascending=True(오름차순))
print(price_per_item.sort_values("item_price",ascending=False)[:10])

# 아이템 가격 분포 차트
item_name_list=price_per_item.index.tolist()
x_pos=np.arange(len(item_name_list)) # x 축
item_price=price_per_item["item_price"].tolist() #y 축

plt.bar(x_pos,item_price,align="center")
plt.ylabel("item price($)")
plt.title("Distribution of item price")
plt.show()

# 아이템 가격 히스토그램

plt.hist(item_price)
plt.ylabel("counts")
plt.title("Histogram of item price")
plt.show()

# 가장 비싼 주문에서 아이템이 총 몇개 팔렸는지 구하기
# chipo.groupby('order_id').sum().sort_values(by='item_price', ascending=False)[:5]

# "Veggie Salad Bowl"이 몇 번 주문되었는지 구하기
# "Veggie Sanlad Bowl"이 몇번 주문되었는지 계산합니다.
chipo_salad=chipo[chipo["item_name"]=="Veggie Salad Bowl"]
# 한 주문 내에서 중복 집계된 item_name을 제거합니다.
chipo_salad=chipo_salad.drop_duplicates(['item_name',"order_id"])

print(len(chipo_salad))
chipo_salad.head(5)

# 'Chicken Bowl'을 2개 이상 주문한 주문 횟수 구하기
# Chicken Bowl 메뉴의 총 주문 수량 구함
chipo_chicken=chipo[chipo['item_name']=="Chiken Bowl"]
chipo_chicken_ordersum=chipo_chicken.groupby('order_id').sum()['quantity']
chipo_chicken_result=chipo_chicken_ordersum[chipo_chicken_ordersum>=2]

print(len(chipo_chicken_result))
print(chipo_chicken_result.head(5))