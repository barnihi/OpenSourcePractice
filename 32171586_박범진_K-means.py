
import numpy as np
import matplotlib.pyplot as plt

dataset = np.array([[ 1.658985,  4.285136],
       [-3.453687,  3.424321],
       [ 4.838138, -1.151539],
       [-5.379713, -3.362104],
       [ 0.972564,  2.924086],
       [-3.567919,  1.531611],
       [ 0.450614, -3.302219],
       [-3.487105, -1.724432],
       [ 2.668759,  1.594842],
       [-3.156485,  3.191137],
       [ 3.165506, -3.999838],
       [-2.786837, -3.099354],
       [ 4.208187,  2.984927],
       [-2.123337,  2.943366],
       [ 0.704199, -0.479481],
       [-0.39237 , -3.963704],
       [ 2.831667,  1.574018],
       [-0.790153,  3.343144],
       [ 2.943496, -3.357075],
       [-3.195883, -2.283926],
       [ 2.336445,  2.875106],
       [-1.786345,  2.554248],
       [ 2.190101, -1.90602 ],
       [-3.403367, -2.778288],
       [ 1.778124,  3.880832],
       [-1.688346,  2.230267],
       [ 2.592976, -2.054368],
       [-4.007257, -3.207066],
       [ 2.257734,  3.387564],
       [-2.679011,  0.785119],
       [ 0.939512, -4.023563],
       [-3.674424, -2.261084],
       [ 2.046259,  2.735279],
       [-3.18947 ,  1.780269],
       [ 4.372646, -0.822248],
       [-2.579316, -3.497576],
       [ 1.889034,  5.1904  ],
       [-0.798747,  2.185588],
       [ 2.83652 , -2.658556],
       [-3.837877, -3.253815],
       [ 2.096701,  3.886007],
       [-2.709034,  2.923887],
       [ 3.367037, -3.184789],
       [-2.121479, -4.232586],
       [ 2.329546,  3.179764],
       [-3.284816,  3.273099],
       [ 3.091414, -3.815232],
       [-3.762093, -2.432191],
       [ 3.542056,  2.778832],
       [-1.736822,  4.241041],
       [ 2.127073, -2.98368 ],
       [-4.323818, -3.938116],
       [ 3.792121,  5.135768],
       [-4.786473,  3.358547],
       [ 2.624081, -3.260715],
       [-4.009299, -2.978115],
       [ 2.493525,  1.96371 ],
       [-2.513661,  2.642162],
       [ 1.864375, -3.176309],
       [-3.171184, -3.572452],
       [ 2.89422 ,  2.489128],
       [-2.562539,  2.884438],
       [ 3.491078, -3.947487],
       [-2.565729, -2.012114],
       [ 3.332948,  3.983102],
       [-1.616805,  3.573188],
       [ 2.280615, -2.559444],
       [-2.651229, -3.103198],
       [ 2.321395,  3.154987],
       [-1.685703,  2.939697],
       [ 3.031012, -3.620252],
       [-4.599622, -2.185829],
       [ 4.196223,  1.126677],
       [-2.133863,  3.093686],
       [ 4.668892, -2.562705],
       [-2.793241, -2.149706],
       [ 2.884105,  3.043438],
       [-2.967647,  2.848696],
       [ 4.479332, -1.764772],
       [-4.905566, -2.91107 ]])

center_x = np.random.uniform(-6,6,4) ## fix me ##
center_y = np.random.uniform(-6,6,4)## fix me ## 
centroids = np.stack([center_x,center_y],axis=-1)

# 예제 데이터셋 시각화

plt.title("The Distribution of Point")
plt.scatter(dataset[:,0],dataset[:,1],label='dataset')
plt.scatter(centroids[:,0],centroids[:,1],
            s=200, label="centroid", marker='+')
plt.legend()
plt.show()

## numpy broadcasting 을 활용해 코드 한줄로 모든 centroids 와 모든 데이터간 거리를 계산하세요. 
dists = np.vstack([np.sqrt(np.sum((dataset-centroids[0:1])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[1:2])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[2:3])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[3:4])**2, axis=1))])

print(dists.shape) # dists 행렬의 형태

cluster_per_point = np.argmin(dists, axis=0)## fix me ##


# 각 군집별로 순회
k = 4 
for i in range(k):    
    # 각 군집에 해당하는 데이터들을 가져옵니다. 
    target_point = dataset[cluster_per_point==i] ## fix me ## 
    #print(target_point)
    
    # 각 군집의 평균을 계산해 centroids 에 할당 합니다. 
    centroids[i] = target_point.mean(axis=0)

num_data = dataset.shape[0]

cluster_per_point = np.ones((num_data))

counter = 0
while True:
    prev_cluster_per_point = cluster_per_point
    
    # (2) 중심점과 각 데이터 사이의 거리를 계산
    dists = np.vstack([np.sqrt(np.sum((dataset-centroids[0:1])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[1:2])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[2:3])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[3:4])**2, axis=1))])## fix me ##
        
    # (3) 각 데이터를 거리가 가장 가까운 군집으로 할당
    ## fix me ##
    cluster_per_point = np.argmin(dists, axis=0)## fix me ##
    # (4) 각 군집 별 점들의 평균을 계산 후, 군집의 중심점을 다시 계산
    ## fix me ##            
    target_point = dataset[cluster_per_point==i] ## fix me ##
    centroids[i] = np.mean(target_point,axis=0) 
    ## (5) cluster 값이 변하지 않으면 while 구문을 종료(np.all 구문 사용)
    if np.all(prev_cluster_per_point == cluster_per_point): ## fix me##:
        break
        
    # 시각화 코드 
    counter += 1
    plt.title("{}th Distribution of Dataset".format(counter))
    for idx, color in enumerate(['r','g','b','y']):
        mask = (cluster_per_point==idx)
        plt.scatter(dataset[mask,0],dataset[mask,1],
                    label='dataset', c=color)
        plt.scatter(centroids[:,0],centroids[:,1],
                    s=200, label="centroid", marker='+')
    plt.show()

def cluster_kmeans(dataset, k):   
       dists = np.vstack([np.sqrt(np.sum((dataset-centroids[0:1])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[1:2])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[2:3])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[3:4])**2, axis=1))])
       print(dists.shape) 
       cluster_per_point = np.argmin(dists, axis=0)
       k = 4 
       for i in range(k):    
           target_point = dataset[cluster_per_point==i] 
           centroids[i] = target_point.mean(axis=0)

       num_data = dataset.shape[0]
       cluster_per_point = np.ones((num_data))
       counter = 0
       while True:
           prev_cluster_per_point = cluster_per_point
           dists = np.vstack([np.sqrt(np.sum((dataset-centroids[0:1])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[1:2])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[2:3])**2, axis=1)),np.sqrt(np.sum((dataset-centroids[3:4])**2, axis=1))])## fix me ##
           cluster_per_point = np.argmin(dists, axis=0)
           target_point = dataset[cluster_per_point==i] 
           centroids[i] = np.mean(target_point,axis=0) 
           if np.all(prev_cluster_per_point == cluster_per_point): 
               break
           
           counter += 1
           plt.title("{}th Distribution of Dataset".format(counter))
           for idx, color in enumerate(['r','g','b','y']):
                  mask = (cluster_per_point==idx)
                  plt.scatter(dataset[mask,0],dataset[mask,1],
                              label='dataset', c=color)
                  plt.scatter(centroids[:,0],centroids[:,1],
                              s=200, label="centroid", marker='+')
           plt.show()
           print(counter)
       return centroids

print(cluster_kmeans(dataset,4))