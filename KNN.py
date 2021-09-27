
import numpy as np
import matplotlib.pyplot as plt
# 0번째 열 : 킥 횟수, 1번째 열 : 키스 횟수
dataset = np.array([
    [3, 104],
    [2, 100],
    [1, 81],
    [101, 10],
    [99, 5],
    [98, 2],
])

labels = np.array(['Romance','Romance','Romance',
          'Action','Action','Action'])

inX = np.array([25, 87])

plt.title("The Category of Movie")
plt.scatter(dataset[:3,0],dataset[:3,1],label='Romance',
            c='g')
plt.scatter(dataset[3:,0],dataset[3:,1],label='Action',
            c='r')
plt.scatter(25,87,label="?",
            c='b')


plt.xlabel('The number of Kick')
plt.ylabel('The number of Kiss')
plt.legend()
#plt.show()


# broadcasting을 이용하면 보다 간결하고 빠른 코드를 작성할 수 있습니다.
## ⚠️⚠️⚠️ 반복문, 순회문 사용하지 않고 numpy broadcasting 을 사용해 구현하세요. ⚠️⚠️⚠️

# 위 정의한 inX와 그 이외의 모든 점들과의 거리를 계산합니다. 
# 거리는 위 정의한 l2 distance 을 사용합니다. 
dists = np.sqrt(np.sum((dataset-inX)**2, axis=1))## fix me ##

print(dists)

## 나와야하는 정답
##  array([ 27.80287755,  26.41968963,  24.73863375, 108.1896483 ,
##      110.45361017, 112.04463396])
#################################
## 나온 정답
## [ 27.80287755  26.41968963  24.73863375 108.1896483  110.45361017
## 112.04463396]


# 파일저장시 해야하는 명령어
#git add 파일명 
#git commit -m "msg" 
#git push opensource master

