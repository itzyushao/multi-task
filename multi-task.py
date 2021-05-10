#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %%capture
# ! pip install pytorch-lightning
# ! pip install pytorch-lightning-bolts
# ! pip install torchvision


# In[2]:


import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.datasets import DummyDataset
from torch.optim import Adam
from torch.utils.data import TensorDataset


# # Multi-Task Learning

# ### Simple Linear Regression

# In[3]:


# Input (temp, rainfall, humidity)
input_1 = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37]], 
                  dtype='float32')
# Targets (apples, oranges)
target_1 = np.array([[52], [84], [115], 
                    [23], [102], [59], 
                    [85], [115], [27], 
                    [101], [51], [82], 
                    [112], [23]], 
                   dtype='float32')

input_1 = torch.from_numpy(input_1)
target_1 = torch.from_numpy(target_1)

train_1 = TensorDataset(input_1, target_1)
train_1[0:3]

model1 = nn.Linear(3, 1)

# Parameters
params = list(model1.parameters())

# Define Loss
import torch.nn.functional as F
loss_fn = F.mse_loss

# Define optimizer
opt = torch.optim.SGD(params, lr=1e-5)

def fit(num_epochs, model1, loss_fn, opt, train_dl1):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb, yb in train_dl1:
            
            # 1. Generate predictions
            pred1 = model1(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred1, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

#fit model for 100 epochs
fit(100, model1, loss_fn ,opt ,train_1)


# In[4]:


list(model1.parameters())


# ### Double-Task Linear Regression

# In[107]:


# Input (temp, rainfall, humidity)
input_1 = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37]], 
                  dtype='float32')
# Targets (apples, oranges)
target_1 = np.array([[52], [84], [115], 
                    [23], [102], [59], 
                    [85], [115], [27], 
                    [101], [51], [82], 
                    [112], [23]], 
                   dtype='float32')

input_2 = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37]], 
                  dtype='float32')
# Targets (apples, oranges)
target_2 = np.array([[52], [84], [115], 
                    [23], [102], [59], 
                    [85], [115], [27], 
                    [101], [51], [82], 
                    [112], [23]], 
                   dtype='float32')

input_1 = torch.from_numpy(input_1)
target_1 = torch.from_numpy(target_1)

train_1 = TensorDataset(input_1, target_1)
train_1[0:3]

input_2 = torch.from_numpy(input_2)
target_2 = torch.from_numpy(target_2)

train_2 = TensorDataset(input_2, target_2)
train_2[0:3]


model1 = nn.Linear(3, 1)
model2 = nn.Linear(3, 1)

##### add another parameters
# Parameters
params = list(model2.parameters()) + list(model1.parameters())

# Define Loss
import torch.nn.functional as F
loss_fn = F.mse_loss

# Define optimizer
opt = torch.optim.SGD(params, lr=1e-5)

def fit(num_epochs, model1, model2, loss_fn, opt, train_dl1, train_dl2):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        ##### add another data
        # Train with batches of data
        for (xb, yb), (xa, ya) in zip(train_dl2 ,train_dl1):
            
            # 1. Generate predictions
            pred1 = model1(xa)
            pred2 = model2(xb)
            
            ##### l1 norm
            regularize_term_1 = sum(p.abs().sum() for p in model1.parameters())
            regularize_term_2 = sum(p.abs().sum() for p in model2.parameters())
            
            ##### edit loss
            # 2. Calculate loss
            loss = loss_fn(pred1, ya)+ loss_fn(pred2, yb) + regularize_term_1+regularize_term_2
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

#fit model for 100 epochs
fit(1000, model1, model2, loss_fn ,opt ,train_1, train_2)


# In[108]:


print(loss_fn(pred1, target_1))
print(loss_fn(pred2, target_2))


# In[109]:


list(model2.parameters())


# In[110]:


list(model1.parameters())


# In[ ]:




