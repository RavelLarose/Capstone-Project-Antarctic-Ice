#!/usr/bin/env python
# coding: utf-8

# In[121]:


#install the necessary packages
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')
get_ipython().system('pip install torch')
get_ipython().system('pip install basemap')


# In[2]:


#Import libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib as mpl
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import torch.optim as optim
import tqdm
import tensorflow as tf
from tensorflow.keras import layers


# In[3]:


#read in the data that was collected in Data Profiling
df = pd.read_csv('IceData.csv', index_col = 0)


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.dtypes


# In[ ]:





# In[ ]:





# In[7]:


#now, define some functions that will help with building this neural network
def DOY_to_Date(year, day):
    Year = str(year)
    Day = str(day)
    res = Year + Day
    result = int(res)
    return result


# In[8]:


#test the DOY_to_Date method
ex = DOY_to_Date(2020, 200)
print(ex)
print(type(ex))


# In[9]:


#use the DOY_to_Date method to convert the dataframe columns from Year and Day of Year to unique values
#df["Unique_Date"] = DOY_to_Date(df["Year"], df["Day of Year"])

l = [0] * len(df)

for row in range(len(df)):
    l[row] = DOY_to_Date(df['Year'][row], df['Day of Year'][row])
    


# In[10]:


#assign the list as the dataframe column 
df.insert(5, "DateValues", l)


# In[11]:


df.head()


# In[12]:


df.drop(columns = "Year", inplace = True)


# In[13]:


df.drop(columns = "Day of Year", inplace = True)


# In[14]:


df.head()


# In[15]:


cols = df.columns.tolist()
cols


# In[16]:


cols = cols[-1:] + cols[:-1]
cols


# In[17]:


df = df[cols]


# In[18]:


df.shape


# In[153]:


miniDF = df.iloc[:200, :]


# In[102]:


column = df['Elevation'].values
column = column.reshape(6737221,1)
column.shape


# In[19]:


#define the halfway point in the dataset
half = 0.5 * len(df)
half + 0.5
int(half)


# In[20]:


#cut the data in half to try and avoid memory issues
df_firstHalf = df.iloc[:int(half), :]
df_secondHalf = df.iloc[int(half)+1:, :]
print(len(df_firstHalf), len(df_secondHalf))


# In[21]:


df_firstHalf


# In[22]:


df_secondHalf


# In[103]:


#scale the data to train more efficiently; first and second halves of the
#data have to be scaled separately
#first half
scale_first = MinMaxScaler() 
scale_first.fit(df_firstHalf)
df_first_scaled = scale_first.transform(df_firstHalf)

#second half
scale_second = MinMaxScaler() 
scale_second.fit(df_secondHalf)
df_second_scaled = scale_second.transform(df_secondHalf)

#also define a scaler for later that will be used to reverse the current scaling
scaler_inverse = MinMaxScaler()
scaler_inverse.fit(column)

print(len(df_first_scaled), len(df_second_scaled))


# In[26]:


df_first_scaled


# In[27]:


df_second_scaled


# In[ ]:





# In[30]:


#Need the data to be in the form [sample, time steps, features (dimension of each element)]
#Tidy up the first half of the data
samples = 10 # Number of samples (in past)
steps = 10 # Number of steps (in future)
X_first = [] # X array
Y_first = [] # Y array
for i in range(df_first_scaled.shape[0] - samples):
    X_first.append(df_first_scaled[i:i+samples, 0:3]) # Independent Samples
    Y_first.append(df_first_scaled[i+samples, 3:]) # Dependent Samples
print('Training Data: Length is ',len(X_first[0:1][0]),': ', X_first[0:1])
print('Testing Data: Length is ', len(Y_first[0:1]),': ', Y_first[0:1])


# In[34]:


X_first


# In[35]:


Y_first


# In[36]:


#Reshape the data so that the inputs will be acceptable to the model.
X_first = np.array(X_first)
Y_first = np.array(Y_first)
print('Dimensions of X', X_first.shape, 'Dimensions of Y', Y_first.shape)


# In[37]:


# # Get the training and testing set
threshold = round(0.9 * X_first.shape[0])
trainX_first, trainY_first = X_first[:threshold], Y_first[:threshold]
testX_first, testY_first =  X_first[threshold:], Y_first[threshold:]
print('Training Length',trainX_first.shape, trainY_first.shape,'Testing Length:',testX_first.shape, testY_first.shape)


# In[38]:


trainX_first


# In[39]:


trainY_first


# In[40]:


# Let's build the GRU
model = Sequential()

# Add a GRU layer with 15 units.
model.add(layers.GRU(15,
                     activation = "tanh",
                     recurrent_activation = "sigmoid",
                     input_shape=(X_first.shape[1], X_first.shape[2])))
# Add a dropout layer (penalizing more complex models) -- prevents overfitting
model.add(layers.Dropout(rate=0.2))


# Add a Dense layer with 1 units
model.add(layers.Dense(1))

# Evaluating loss function of MSE using the adam optimizer.
model.compile(loss='mean_squared_error', optimizer = 'adam')

# Print out architecture.
model.summary()


# In[107]:


# Fitting the data
history = model.fit(trainX_first,
                    trainY_first,
                    shuffle = False, # Since this is time series data
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1) # Verbose outputs data


# In[108]:


# Plotting the loss iteration
training_loss_first = history.history['loss']
test_loss_first = history.history['val_loss']
plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['val_loss'], label ='validation loss')
plt.legend()
# Note:
# if training loss >> validation loss -> Underfitting
# if training loss << validation loss -> Overfitting (i.e model is smart enough to have mapped the entire dataset..)
# Several ways to address overfitting:
# Reduce complexity of model (hidden layers, neurons, parameters input etc)
# Add dropout and tune rate
# More data :)


# In[109]:


y_pred_first = model.predict(testX_first)
plt.plot(testY_first, label = 'True Value')
plt.plot(y_pred_first, label = 'Forecasted Value')
plt.legend()


# In[110]:


#do the same to the back end of the dataset



# In[111]:


#Need the data to be in the form [sample, time steps, features (dimension of each element)]
samples = 10 # Number of samples (in past)
steps = 10 # Number of steps (in future)
X_second = [] # X array
Y_second = [] # Y array
for i in range(df_second_scaled.shape[0] - samples):
    X_second.append(df_second_scaled[i:i+samples, 0:3]) # Independent Samples
    Y_second.append(df_second_scaled[i+samples, 3:]) # Dependent Samples
print('Training Data: Length is ',len(X_second[0:1][0]),': ', X_second[0:1])
print('Testing Data: Length is ', len(Y_second[0:1]),': ', Y_second[0:1])


# In[112]:


#Reshape the data so that the inputs will be acceptable to the model.
X_second = np.array(X_second)
Y_second = np.array(Y_second)
print('Dimensions of X', X_second.shape, 'Dimensions of Y', Y_second.shape)


# In[113]:


# # Get the training and testing set
threshold = round(0.9 * X_second.shape[0])
trainX_second, trainY_second = X_second[:threshold], Y_second[:threshold]
testX_second, testY_second =  X_second[threshold:], Y_second[threshold:]
print('Training Length',trainX_second.shape, trainY_second.shape,'Testing Length:',testX_second.shape, testY_second.shape)


# In[114]:


# Fitting the data
history = model.fit(trainX_second,
                    trainY_second,
                    shuffle = False, # Since this is time series data
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1) # Verbose outputs data


# In[115]:


# Plotting the loss iteration
training_loss_second = history.history['loss']
test_loss_second = history.history['val_loss']
plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['val_loss'], label ='testing loss')
plt.legend()


# In[116]:


# This is a one step forecast (based on how we constructed our model)
y_pred_second = model.predict(testX_second)
plt.plot(testY_second, label = 'True Value')
plt.plot(y_pred_second, label = 'Forecasted Value')
plt.legend()


# In[117]:


#reverse scaling of the data with the inverse scaler we defined before
y_pred_first = scaler_inverse.inverse_transform(y_pred_first)
y_pred_second = scaler_inverse.inverse_transform(y_pred_second)


# In[118]:


#display both readouts in a single figure
y_pred = np.append(y_pred_first, y_pred_second)
testY = np.append(testY_first, testY_second)
plt.plot(testY, label = 'True Value')
plt.plot(y_pred, label = 'Forecasted Value')
fig = plt.figure()
fig.set_figwidth(15)
fig.set_figheight(7)
plt.plot(testY, label = 'True Value')
plt.plot(y_pred, label = 'Forecasted Value')
plt.xlabel('Network Timestep')
plt.ylabel('Ice Elevations')
plt.plot
plt.legend()


# In[119]:


train_loss = np.append(training_loss_first, training_loss_second)
test_los = np.append(test_loss_first, test_loss_second)
plt.plot(train_loss, label = 'training loss')
plt.plot(test_loss, label ='testing loss')
plt.xlabel('Network Timestep')
plt.ylabel('Mean Squared Error')
plt.legend()


# In[ ]:


#now that the model is working, feed it some projections without a corresponding answer set and see what it comes up with


# In[125]:


#plot the data we have across antarctica
#first plot antarctica
from mpl_toolkits.basemap import Basemap
m = Basemap(projection='spstere',boundinglat=-60,lon_0=180,resolution='c')

#draw the coast, parallel lines and meridian lines, then fill in the ocean for aesthetics
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.fillcontinents(color = 'white')
m.drawmapboundary(fill_color='aqua')

#draw the longitude and latitude values of the existing data
lons = df['Longitude'].values
lats = df['Latitude'].values
m.scatter(lons, lats, marker = 'o', color='b', zorder=10, latlon = True)


# In[147]:


#plot the true longitude and latitude elevation projections across the map
#get the longitude and latitude values that match the projected variables
longs_proj = testX_first[:, 1]
longs_proj = np.append(longs_proj, testX_second[:, 1])
lats_proj = testX_first[:, 2]
lats_proj = np.append(lats_proj, testX_second[:, 1])

r = testY_first
r = np.append(testY_first, testY_second)
r = pd.cut(r, 5)

m = Basemap(projection='spstere',boundinglat=-60,lon_0=180,resolution='c')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.fillcontinents(color = 'white')
m.drawmapboundary(fill_color='aqua')
m.scatter(longs_proj, lats_proj, marker = 'o', zorder=10, latlon = True)


# In[ ]:


#and then plot the model's predictions


# In[145]:


r = testY_first
r = np.append(testY_first, testY_second)
r = pd.cut(r, 5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




