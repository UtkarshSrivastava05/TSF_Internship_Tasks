# Task 3 - To Explore Unsupervised Machine Learning

# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Download the iris dataset
url = "https://drive.google.com/file/d/11Iq7YvbWZbt8VXjfm06brx66b10YiwK-/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
iris = pd.read_csv(path)

# Explore the dataset
iris.head() # See the first 5 rows

# Correlation
corr_df = iris.corr()
corr_df


# Heat map
plt.rcParams['figure.figsize'] = [10, 8]
sns.heatmap(iris.corr(),annot= True )


# Count of flowers in each unique species
iris['Species'].value_counts()

# Dependent variables array
X = iris.iloc[:, 0:4].values


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    
    # We will "k-means++" initialization method to avoid falling into the random initialization trap"
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)    
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    

# Plotting the the elbow method graph
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.grid()
plt.show()


# Fitting K-Means on the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# Visualising the clusters on the first 2 columns with centroids
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolor')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.legend()
plt.grid()
plt.show()