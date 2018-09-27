# 1. IMPORTING THE LIBRARIES:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# 2. IMPORTING THE DATASET:
dataset = pd.read_csv('C:\\Users\\1385\\Desktop\\Credit_Card_Applications.csv')
#split the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#the class column: 1 the application of the credit card is approved and 0 is not approved
# in the self organizing map we can clearly distinguish the customers who didn't get approvals to their application and the customer who got approval
# this will be useful to detect the fraudulent customers who got approval





dataset.head()




#FEATURE SCALING:
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
#feature range = the range of our scaled input values = normalization
X = sc.fit_transform(X)



#TRAINING THE SOM:
from minisom import MiniSom

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
#we initialize the weights

som.train_random(data = X, num_iteration = 100)
#we train the algorithm




#VISUALIZING THE RESULTS:
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

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

#FINDING THE FRAUDS:
mappings = som.win_map(X)
len(mappings[(7,1)])
frauds=np.concatenate((mappings[(7,1)], mappings[(7,2)], ), axis = 0)
frauds=sc.inverse_transform(frauds)
frauds_df =pd.DataFrame(frauds)
frauds_df

mappings[(4,7)]

