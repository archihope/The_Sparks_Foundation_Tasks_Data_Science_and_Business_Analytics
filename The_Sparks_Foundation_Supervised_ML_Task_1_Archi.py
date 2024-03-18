import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Reading the Data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("----Data uploaded Successfully----")

# ----Data uploaded Successfully----#

print(data.head())

#   Hours  Scores
#0    2.5      21
#1    5.1      47
#2    3.2      27
#3    8.5      75
#4    3.5      30

print(data.isnull == True)

# False

sns.set_style('darkgrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()

# Image Marks Vs Study Hours

sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())

# Image Regression Plot


X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")

pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})

compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})


plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))

hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))

