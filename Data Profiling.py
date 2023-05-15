#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
from pathlib import Path


# In[75]:


#first, we want to read the data into DataFrames one .txt file at a time
#we'll put all the data into a single dataframe, called iceData, with columns
#year ; day of year ; longitude ; latitude ; elevation

#first define the data frame and the columns to be included 
iceData = pd.DataFrame(columns = ['Year', 'Day of Year', 'Longitude', 'Latitude', 'Elevation'])


# In[ ]:





# In[76]:


#we will loop through each folder one at a time, starting with 2008 as a test 
#(this is easy, as there is only 1 file)

i = 0
folder = Path('C:\\Users\\LAROS\\1Capstone Work\\Data\\2008')


#for each file in the 2008 folder
for child in folder.iterdir():
    if child.is_file():
        #open the file
        with open(child) as file: 
            #go through each row in the file
            for line in file:
                #if the line starts with '#', skip it
                lineStrings = line.split(' ')
                if not line:
                    continue
                elif lineStrings[0] == '#':
                    continue
                else:
                    #if the line doesn't start with #, then it is formatted as year, doy, sod, lon, lat, elev
                    #read year, doy, lon, lat, elev into the dataframe
                    iceData.loc[i] = [lineStrings[0], lineStrings[1], lineStrings[3], lineStrings[4], lineStrings[5]]
                    
                    #and increment the index
                    i+=1
            file.close();


# In[77]:


#test to make sure the dataframe is being read to
iceData.head()


# In[78]:


iceData.shape


# In[79]:


#repeat the process, this time calling the 2009 folder
folder = Path('C:\\Users\\LAROS\\1Capstone Work\\Data\\2009')


#for each file in the 2008 folder
for child in folder.iterdir():
    if child.is_file():
        #open the file
        with open(child) as file: 
            #go through each row in the file
            for line in file:
                #if the line starts with '#', skip it
                lineStrings = line.split(' ')
                if not line:
                    continue
                elif lineStrings[0] == '#':
                    continue
                else:
                    #if the line doesn't start with #, then it is formatted as year, doy, sod, lon, lat, elev
                    #read year, doy, lon, lat, elev into the dataframe
                    iceData.loc[i] = [lineStrings[0], lineStrings[1], lineStrings[3], lineStrings[4], lineStrings[5]]
                    
                    #and increment the index
                    i+=1
            print("Completed work on ", child)
            file.close();


# In[80]:


iceData.shape


# In[81]:


iceData.head()


# In[84]:


#repeat the process with 2010
folder = Path('C:\\Users\\LAROS\\1Capstone Work\\Data\\2010')

iceData2010 = pd.DataFrame(columns = ['Year', 'Day of Year', 'Longitude', 'Latitude', 'Elevation'])

#for each file in the 2010 folder
for child in folder.iterdir():
    if child.is_file():
        #open the file
        with open(child) as file: 
            #go through each row in the file
            for line in file:
                #if the line starts with '#', skip it
                lineStrings = line.split(' ')
                if not line:
                    continue
                elif lineStrings[0] == '#':
                    continue
                else:
                    #if the line doesn't start with #, then it is formatted as year, doy, sod, lon, lat, elev
                    #read year, doy, lon, lat, elev into the dataframe
                    iceData2010.loc[i] = [lineStrings[0], lineStrings[1], lineStrings[3], lineStrings[4], lineStrings[5]]
                    
                    #and increment the index
                    i+=1
            print("Completed work on ", child)
            file.close();


# In[85]:


#repeat the process with 2011
folder = Path('C:\\Users\\LAROS\\1Capstone Work\\Data\\2011')

iceData2011 = pd.DataFrame(columns = ['Year', 'Day of Year', 'Longitude', 'Latitude', 'Elevation'])

#for each file in the 2008 folder
for child in folder.iterdir():
    if child.is_file():
        #open the file
        with open(child) as file: 
            #go through each row in the file
            for line in file:
                #if the line starts with '#', skip it
                lineStrings = line.split(' ')
                if not line:
                    continue
                elif lineStrings[0] == '#':
                    continue
                else:
                    #if the line doesn't start with #, then it is formatted as year, doy, sod, lon, lat, elev
                    #read year, doy, lon, lat, elev into the dataframe
                    iceData2011.loc[i] = [lineStrings[0], lineStrings[1], lineStrings[3], lineStrings[4], lineStrings[5]]
                    
                    #and increment the index
                    i+=1
            print("Completed work on ", child)
            file.close();


# In[86]:


#repeat the process with 2012
folder = Path('C:\\Users\\LAROS\\1Capstone Work\\Data\\2012')

iceData2012 = pd.DataFrame(columns = ['Year', 'Day of Year', 'Longitude', 'Latitude', 'Elevation'])

#for each file in the 2008 folder
for child in folder.iterdir():
    if child.is_file():
        #open the file
        with open(child) as file: 
            #go through each row in the file
            for line in file:
                #if the line starts with '#', skip it
                lineStrings = line.split(' ')
                if not line:
                    continue
                elif lineStrings[0] == '#':
                    continue
                else:
                    #if the line doesn't start with #, then it is formatted as year, doy, sod, lon, lat, elev
                    #read year, doy, lon, lat, elev into the dataframe
                    iceData2012.loc[i] = [lineStrings[0], lineStrings[1], lineStrings[3], lineStrings[4], lineStrings[5]]
                    
                    #and increment the index
                    i+=1
            print("Completed work on ", child)
            file.close();


# In[87]:


#repeat the process with 2013
folder = Path('C:\\Users\\LAROS\\1Capstone Work\\Data\\2013')

iceData2013 = pd.DataFrame(columns = ['Year', 'Day of Year', 'Longitude', 'Latitude', 'Elevation'])

#for each file in the 2008 folder
for child in folder.iterdir():
    if child.is_file():
        #open the file
        with open(child) as file: 
            #go through each row in the file
            for line in file:
                #if the line starts with '#', skip it
                lineStrings = line.split(' ')
                if not line:
                    continue
                elif lineStrings[0] == '#':
                    continue
                else:
                    #if the line doesn't start with #, then it is formatted as year, doy, sod, lon, lat, elev
                    #read year, doy, lon, lat, elev into the dataframe
                    iceData2013.loc[i] = [lineStrings[0], lineStrings[1], lineStrings[3], lineStrings[4], lineStrings[5]]
                    
                    #and increment the index
                    i+=1
            print("Completed work on ", child)
            file.close();


# In[88]:


#remove some values that may have gotten roped into the wrong dataframe
iceData = iceData[iceData.Year != 2010]


# In[89]:


iceDataAll = pd.concat([iceData, iceData2010, iceData2011, iceData2012, iceData2013], ignore_index=True, sort=False)


# In[90]:


iceDataAll.shape


# In[91]:


iceDataAll.to_csv("All Ice Data", sep=',')


# In[ ]:




