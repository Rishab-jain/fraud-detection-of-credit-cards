
# coding: utf-8

# In[1]:


# 1. IMPORTING THE LIBRARIES:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


# 2. IMPORTING THE DATASET:
#data=pandas.read_csv('C:\\Users\\1385\\Desktop\\context_table_aegis (1).csv',sep=",")
dataset = pd.read_csv('C:\\Users\\1385\\Desktop\\Credit_Card_Applications.csv')
#split the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#y = the class column: 1 the application of the credit card is approved and 0 is not approved
# in the self organizing map we can clearly distinguish the customers who didn't get approvals to their application and the customer who got approval
# this will be useful to detect the fraudulent customers who got approval
y


# In[3]:


dataset.head()
#the customers are the inputs of our neural network


# In[4]:


#FEATURE SCALING:
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
#feature range = the range of our scaled input values = normalization
X = sc.fit_transform(X)


# In[5]:


#TRAINING THE SOM:
#this code made by a developer, an implementation of SOM:
from minisom import MiniSom

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
#x and y are the dimensions: we can choose whatever dimensions we want
#10 by 10 grid
#input lane: the number of features we have in the X. 14 attributes + customer ID
#sigma: is the radius

som.random_weights_init(X)
#we initialize the weights

som.train_random(data = X, num_iteration = 100)
#we train the algorithm


# In[6]:


#VISUALIZING THE RESULTS:
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
#bone function: if we execute it alone, we get a white window without anything in it
#because we just initialize the window that will contain the map
#pcolor: different colors are corresponding to the different rane values of the mean interneuron distances
#distance map method: will return all the mean interneuron distances in one matrix
#.T: transpose of this MID matrix
#colorbar: adds which color refers to highest MIDs and lowest MIDs. these are the normalized values
#frauds are where the MID is the highest, the outlaying winning nodes, so here where the color is lighter
#markers: o is circle, s is square: for customers who got approval and who didnt
#red circles: customer who didn't get approval
#green squares: got approval

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
# i: is the different values of all the indexes in X 0 to 689
# x: is the different vectors of customers. rows
#enumerate: it allows to loop over something and have an automatic counter
# w: is the winning node. we get the winning node for specific customer
# we want to add a colored market on the winning node
#the marker depends on whether the customer get approval or not
# w[0], w[1]: that's the coordinates of the winning node. 0.5 puts the markers on the center
# markers[y[i]]: this will give us the information whether the customer get approval or not
# if the y[i] = 0 then the customer didn't get approval and then
#markers[0] will be the red circle and so on. 
#it's same for the colors[y[i]]


# In[ ]:


#FINDING THE FRAUDS:
mappings = som.win_map(X)
len(mappings[(7,1)])
frauds=np.concatenate((mappings[(7,1)], mappings[(7,2)], ), axis = 0)
frauds=sc.inverse_transform(frauds)
frauds_df =pd.DataFrame(frauds)
frauds_df
#column number 0 is the customer ID.
#the bank can investigate more about those customers to catch potential fraud


# In[31]:


mappings[(4,7)]

