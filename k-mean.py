import numpy as np
def kmeans(data, k,max_iters=10):
    centers = data[np.random.choice(data.shape[0],k,replace=False)]
    for _ in range(max_iters):
        clusters = [[]for _ in range(k)]
        for point in data:
            distances = np.linalg.norm(centers - point,axis=1)
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(point)
        new_centers = np.array([np.mean(cluster,axis=0) for cluster in clusters if cluster])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return clusters,centers
data = np.array([
    [1.0,2.0],
    [1.5,1.8],
    [5.0,8.0],
    [8.0,8.0],
    [1.0,0.6],
    [9.0,11.0],
    [8.0,2.0],
    [10.0,2.0],
    [9.0,3.0]
])

clusters,centers = kmeans(data,k=3)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i}:")
    print(np.array(cluster))
    print(f"Center: {centers[i]}")

    
