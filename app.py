from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from Kmeans import Kmeans
import pandas as pd

#centroids = [(-5,-5),(5,5)]
#cluster_std = [1,1]

# X, y = make_blobs(
#    centers=centroids,
#    n_features=2,
#    random_state=2
# )
# plt.scatter(X[:,0],[:,1])

df = pd.read_csv('student_clustering.csv')

X = df.iloc[:,:].values

Km =Kmeans(n_clusters=2,max_iter=100)
y_means = Km.fit_predict(X)


plt.scatter(X[y_means == 0,0], X[y_means == 0,1],color='red')
plt.scatter(X[y_means == 1,0], X[y_means == 1,1],color='blue')
plt.scatter(X[y_means == 2,0], X[y_means == 2,1],color='green')
plt.scatter(X[y_means == 3,0], X[y_means == 3,1],color='yellow')
plt.show()

# X, y = make_blobs(
#     n_samples=100,
#     centers=2,
#     random_state=42
# )

# wcss = []

# for k in range(1,11):

#     km = Kmeans(n_clusters=k)

#     cluster_group = km.fit_predict(X)

#     wcss.append(
#         km.get_wcss(X,cluster_group)
#     )

# plt.plot(range(1,11), wcss, marker='o')
# plt.xlabel("Number of Clusters")
# plt.ylabel("WCSS")
# plt.title("Elbow Method")
# plt.show()
