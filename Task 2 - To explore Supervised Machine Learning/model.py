# Task 2 - To Explore Supervised Machine Learning

# Importing all the libraries required for this task
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  


# Downloading the dataset and exploring it
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)
print("Dataset downloaded.")

data.head(10)

data.describe()


# Visualizing the dataset
# Plotting the distribution of scores
data.plot(x = 'Hours', y = 'Scores', style = 'o', c = 'red', figsize = (10,8))
plt.title('Hours vs Percentage') 
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
plt.grid()
plt.show()


# Preparing the data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# Training the Model
# Fitting Simple Linear Regresstion to the Training set
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

# Plotting the regression line
line = (regressor.coef_ * X) + regressor.intercept_

# Plotting for the test data
plt.figure(figsize=(10,8))
plt.scatter(X, y, color = 'red')
plt.plot(X, line, color = 'blue');
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
plt.grid()
plt.show()


# Making Predictions
# Testing data (in Hours)
print("The training test set: ") 
print(X_test)

# Predicting the scores
y_pred = regressor.predict(X_test)

# Comparing Actual vs Predicted value of the score
df = pd.DataFrame({'Actual Score': y_test, 'Predicted Score': y_pred})  
df

hours = 9.25
prediction = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("The  predicted Score of the student = {} %".format(round(prediction[0], 2)))  # Rounding the predicted score upto 2 decimal places


# Evaluating the model
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) 
print('Root Mean Absolute Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))


# Saving model to disk
import pickle
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[9.25]]))