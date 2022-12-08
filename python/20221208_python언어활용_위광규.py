#주어진 자연수가 홀수인지 짝수인지 판별해주는 함수(is_odd)를 작성해보자
def is_odd(number):
    if number%2==0:
        return True # 주어진 자연수가 2로 나누었을때 0으로 딱 떨어지면 짝수이므로 True
    else :
        return False # 주어진 자연수가 2로 나누었을때 0이 안되면 홀수이므로 False

print(is_odd(4))

# python 언어 활용
