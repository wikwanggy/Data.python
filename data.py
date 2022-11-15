# 판다스 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 판다스의 데이터 프레임을 생성하기.
names=["Bob","Jessica","Mary","John","Mel"]
births=[965,155,77,578,973]
custom=[1,5,25,13,23232]

BabyDataSet = list(zip(names,births)) # names와 births의 데이터를 엮어서 하나의 리스트에 저장.
df = pd.DataFrame(data=BabyDataSet,columns=['Names','Births']) # 데이터를 엑셀 형태로 저장하고 열 이름은 각각 Names Births로 저장



arr1=np.arange(15).reshape(3,5)

print(arr1)
print(arr1.shape)
print(arr1.dtype)
arr3=np.zeros((3,4))
print(arr3)

arr4=np.array([[1,2,3],[4,5,6]],dtype=np.float64)
arr5=np.array([[7,8,9],[10,11,12]],dtype=np.float64)

print(arr4+arr5)  


y=df['Births']
x=df['Names']

# 막대 그래프를 출력합니다
plt.bar(x,y) # 막대 그래프 객체 생성
plt.xlabel('Names') #x축 제목
plt.ylabel('Births') #y축 제목
plt.title('Bar plot') # 그래프 제목
plt.show() #그래프 출력

# 랜덤 추출 시드를 고정합니다.
np.random.seed(19920613)

# 산점도 데이터를 생성합니다.
x=np.arange(0.0,100.0,5.0) # 시작점 0(0은 생략 가능)에서 100(끝 점 100)까지 5씩 증가(1은 생략)
y=(x+1.5)+ np.random.rand(20)*50

#산점도 데이터를 출력합니다
#          x값,y값,색상,색상 투명도,범례
plt.scatter(x,y,c="b",alpha=0.5,label="scatter point") 
plt.xlabel("x") # x축 제목
plt.ylabel("y") # y축 제목
plt.legend(loc='upper left')
plt.title('Scatter plot')
plt.show()
