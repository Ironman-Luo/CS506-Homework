#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# In[2]:


z = np.random.random(50)*2 + 2
x1 = np.append((np.random.random(25) * -4),(np.random.random(25) * 3))
y1 = x1 + z
x2 = np.append((np.random.random(25) * -4),(np.random.random(25) * 3))
y2 = x2 - z

labels = [1] * 50 + [0] * 50
X = np.append(x1,x2)
Y = np.append(y1,y2)
data = pd.DataFrame({"x":X, "y":Y, "labels": labels}).to_numpy()
linearR = LinearRegression().fit( data[:,[0,1]], data[:,[2]])
plot_x = np.linspace(-4,4)
plot_y = [(0.5 - linearR.intercept_[0] - linearR.coef_[0,0] * i)/linearR.coef_[0,1] for i in plot_x]

logisticR = LogisticRegression().fit(data[:,[0,1]], data[:,2])
b = logisticR.intercept_[0]
w1, w2 = logisticR.coef_.T
c = -b/w2
m = -w1/w2
xd = np.array([-4, 4])
yd = m*xd + c

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.suptitle('data_without_outlier   vs     data_with_outlier')
axes[0].scatter(x1,y1, c = 'red', marker = "x")
axes[0].scatter(x2,y2, c = 'blue', marker = "o")
axes[0].plot(plot_x, plot_y, c = 'red', label = "linearR")
axes[0].plot(xd, yd, c = "blue", label = "logisticR")
axes[0].legend()
axes[0].axis([-4, 10, -20, 7]) 


outliers_x = np.random.random(5)*3 + 6
outliers_y = np.random.random(5)*3 - 18

z = np.random.random(50)*2 + 2
x1 = np.append((np.random.random(25) * -4),(np.random.random(25) * 3))
y1 = x1 + z
x2 = np.append((np.random.random(25) * -4),(np.random.random(25) * 3))
y2 = x2 - z

labels = [1] * 50 + [0] * 55
X = np.append(x1,x2)
Y = np.append(y1,y2)
X = np.append(X,outliers_x)
Y = np.append(Y, outliers_y)
data = pd.DataFrame({"x":X, "y":Y, "labels": labels}).to_numpy()
linearR = LinearRegression().fit( data[:,[0,1]], data[:,[2]])
plot_x = np.linspace(-4,4)
plot_y = [(0.5 - linearR.intercept_[0] - linearR.coef_[0,0] * i)/linearR.coef_[0,1] for i in plot_x]

logisticR = LogisticRegression().fit(data[:,[0,1]], data[:,2])
b = logisticR.intercept_[0]
w1, w2 = logisticR.coef_.T
c = -b/w2
m = -w1/w2
xd = np.array([-4, 4])
yd = m*xd + c

axes[1].scatter(x1,y1, c = 'red', marker = "x")
axes[1].scatter(np.append(x2,outliers_x), np.append(y2,outliers_y), c = 'blue', marker = "o")
axes[1].plot(plot_x, plot_y, c = 'red', label = "linearR")
axes[1].plot(xd, yd, c = "blue", label = "logisticR")
axes[1].legend()
axes[1].axis([-4, 10, -20, 7]) 
fig.savefig('p1c.png')

