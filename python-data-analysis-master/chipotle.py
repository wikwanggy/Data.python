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