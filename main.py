#IMPORTING LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn

#IMPORTING DATASET
dataset=pd.read_csv('https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv')

#Analysing details
dataset.head()
dataset.tail()
dataset.describe()

#dimension of dataset
print('There are',dataset.shape[0],'rows and',dataset.shape[1],'columns in the dataset')

#check for repated values
print('There are',dataset.duplicated().sum(),'duplicate items in the dataset')

#Check for NULL values
dataset.isnull().sum()

#Schema of dataset
dataset.info()

#correlating dataset using corr function
c=dataset.corr()
c

#EDA on dataset
#correlatiom matrix
sns.heatmap(c,annot=True,cmap='Blues')
plt.show()

#outlier detection in target variables
outliers = ['Profit']
plt.rcParams['figure.figsize'] = [8,8]
sns.boxplot(data=dataset[outliers], orient="v", palette="Set2" , width=0.7) # orient = "v" : vertical boxplot , 
                                                                            # orient = "h" : hotrizontal boxplot
plt.title("Outliers Variable Distribution")
plt.ylabel("Profit Range")
plt.xlabel("Continuous Variable")

plt.show()

#state-wise outlier detection
sns.boxplot(x = 'State', y = 'Profit', data = dataset)
plt.show()

#histogram on profit
sns.distplot(dataset['Profit'],bins=5,kde=True)
plt.show()

#pair plot
sns.pairplot(dataset)
plt.show()

#model developement
# spliting Dataset in Dependent & Independent Variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,  4]. values

#label encoder
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X1 = pd.DataFrame(X)
X1.head()

#splitting dataset into training and testing data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)
x_train

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)
print('Model has been trained successfully')

#testing model using predict function
y_pred = model.predict(x_test)
y_pred

#testing scores
testing_data_model_score = model.score(x_test, y_test)
print("Model Score/Performance on Testing data",testing_data_model_score)

training_data_model_score = model.score(x_train, y_train)
print("Model Score/Performance on Training data",training_data_model_score)

#comparing predicted and actual parameters
df = pd.DataFrame(data={'Predicted value':y_pred.flatten(),'Actual Value':y_test.flatten()})
df

#Model evaluation
from sklearn.metrics import r2_score

r2Score = r2_score(y_pred, y_test)
print("R2 score of model is :" ,r2Score*100)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_pred, y_test)
print("Mean Squarred Error is:" ,mse*100)

rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Root Mean Squarred Error is : ",rmse*100)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_pred,y_test)
print("Mean Absolute Error is:" ,mae)
