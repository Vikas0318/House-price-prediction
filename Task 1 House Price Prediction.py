#!/usr/bin/env python
# coding: utf-8

# # Task 1 : Predict House Price

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Stept 1 : Load the dataset


# In[3]:


data = pd.read_csv("C:\\Users\\Vikas\\OneDrive\\Documents\\Python Data Files\\kc_house_data.csv")


# In[4]:


data.head()


# In[5]:


#Step 2 : Select relevant features


# In[6]:


features = ['sqft_living','bedrooms']


# In[7]:


x=data[features]
y=data['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[8]:


#Step 3 : Visualizing the data


# In[9]:


plt.scatter(x['sqft_living'],y)
plt.xlabel('Square Foot')
plt.ylabel('Price')
plt.title('Relationship between Square Foot and Price')
plt.show()


# In[10]:


# Step 4 : Assign feature names to x_train and x_test


# In[11]:


x_train = pd.DataFrame(x_train, columns=features)
x_test = pd.DataFrame(x_test, columns=features)


# In[12]:


# Step 5 : Choose the regression model


# In[13]:


model = LinearRegression()


# In[14]:


# Step 6 : Train the model


# In[15]:


model.fit(x_train, y_train)


# In[16]:


# Step 7 : Evaluate the model


# In[17]:


y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse}")


# In[18]:


# Step 8 : Desplay the model
# Use the trained model to predict prices for new houses


# In[19]:


new_house = [[2000,4]]  # Example input for sqft_living and bedrooms
predicted_price = model.predict(new_house)
print(f"Predicted Price for new house : {predicted_price}")

