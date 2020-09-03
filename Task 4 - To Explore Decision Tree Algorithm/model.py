# Task 4 - To Explore Decision Tree Algorithm

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

# Count of flowers in each unique species
iris['Species'].value_counts()

# Correlation
corr_df = iris.corr()
corr_df

# Plotting pairwise relationships of the dataset.
plt.rcParams['figure.figsize'] = [10, 8]
sns.pairplot(iris.dropna(),hue="Species")

# Plot of Sepal's Length Vs. Width
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal's Length")
fig.set_ylabel("Sepal's Width")
fig.set_title("Sepal's Length Vs. Width")
fig=plt.gcf()
plt.show()

# Plot of Petal's Length Vs. Width
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal's Length")
fig.set_ylabel("Petal's Width")
fig.set_title("Petal's Length Vs. Width")
fig=plt.gcf()
plt.show()


# Splitting the dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X = iris.iloc[:, 1:5].values
y = iris.iloc[:, 5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Fitting the classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating the accuracy of our model
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('Accuracy of this model is {}%'.format(round(acc*100,2)))


# Visualizing the Decision Tree Classifier
from sklearn.tree import plot_tree

f_name = X.tolist()
col_name = y.tolist()
plt.figure(figsize=(25,20))
tree_image = plot_tree(classifier, feature_names = f_name, class_names = col_name ,max_depth = 5, precision = 4, label = "all", 
                       filled = True, rounded = True)
plt.savefig('Tree_image')


# Saving the model
import pickle
pickle.dump(classifier, open('model.pkl','wb'))

# Loading the model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5.5,2.3,4.0,1.3]]))