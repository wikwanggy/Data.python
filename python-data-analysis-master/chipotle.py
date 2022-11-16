import pandas as pd

file_path="python-data-analysis-master/data/chipotle.tsv"
#read.csv() 함수로 데이터를 데이터 프레임 형태(엑셀)로 불러옵니다.
chipo=pd.read_csv(file_path, sep="\t")

print(chipo.shape)
print("--------------------------------")
print(chipo.info())

chipo.head(10)

print(chipo.columns)
print("--------------------------------")
print(chipo.index)