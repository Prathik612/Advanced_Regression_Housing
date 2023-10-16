import torch
from torch import nn
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}\n")

train_data = pd.read_csv("house-prices-advanced-regression-techniques\\train.csv")

#print(train_data.head)

print(train_data.isnull().sum().to_string())
print(train_data.shape)

#plt.figure(figsize = (8,6))
#sb.heatmap(train_data.isnull(), cbar=False , cmap = 'magma')
#plt.show()

def cleanData(data = train_data):
    data = data.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)

    return data

train_data = cleanData()
print("New: \n")
print(train_data.isnull().sum().to_string())
print(train_data.shape)