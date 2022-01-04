#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with Python
# 
# In this section we will see how the Python jupter notebook library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# ## Prediction using Supervised ML-
# 
# ## Simple Linear Regression
# 
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.
# 
# # The sparks Foundation
#  Internship :- Data science And Business Analytics
# 
# Batch :- GRIPJAN22 
#  
#  Task 1 :- Prediction using Supervised ML
#  
#  Author:- Sanjeev Singh
#  

# In[1]:


#importing all the libraries.
import pandas as pd 
import matplotlib as mpl
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression 
from scipy import stats
import seaborn as sns
import numpy as np


# In[2]:


#importing and reaading the data
data = pd.read_excel('E:\PYTHON\students_scores.xlsx')


# In[3]:


data


# In[4]:


import matplotlib.pyplot as plt 
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel(' Score')  
plt.show()


# In[5]:


data.shape


# In[6]:


data.describe()


# ## Train-Test Split
# 
# Train/Test is a method to measure the accuracy of your model. It is called Train/Test because you split the the data set into two sets: a training set and a testing set.

# In[7]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 


# In[8]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.3, random_state=0) 


# In[9]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# In[10]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line, color='red');
plt.show()


# ## Making Predictions
# Now that we have trained our algorithm, it's time to make some predictions.

# In[11]:


print(y_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[12]:


print(y_pred)


# In[13]:


#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'yellow')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs. Percentage (Training set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage of marks')
plt.show()


# In[14]:


#Visualising the Test set results
plt.scatter(X_test, y_test, color = 'yellow')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs. Percentage (Test set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage of marks')
plt.show()


# In[15]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[16]:


#predicting the score 
dataset = np.array(9.25)
dataset = dataset.reshape(-1, 1)
pred = regressor.predict(dataset)
print("If the student studies for 9.25 hours/day, the score is {}.".format(pred))


# ## Error metrics
# 
# 

# In[17]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# ## conclusion 
# if student studies for 9.25 hours/day ,so predicted score will be 92.915

# In[ ]:





# In[ ]:




