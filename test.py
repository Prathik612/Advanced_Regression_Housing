import torch
from torch import nn
from torch import optim
import pandas as pd
import numpy as np
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}\n")

test_data = pd.read_csv("house-prices-advanced-regression-techniques\\test.csv")
#print(test_data)

def cleanData(data = test_data):
    data = data.drop(columns=["Id","Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)

    #y = data["SalePrice"]
    #data = data.drop(columns=["SalePrice"])

    num_cols = data.select_dtypes(np.number).columns
    cat_cols = data.select_dtypes(include=['object']).columns

    data_cat = pd.get_dummies(data[cat_cols], dummy_na=True)
    data_num = data[num_cols].apply(lambda x: (x - x.mean()) / (x.std()))
    #data_num = data_num.fillna(0)
    mean = data_num.mean(axis=0)
    data_num = data_num.fillna(mean)

    data = pd.concat([data_num, data_cat], axis=1)

    return data

X_data = cleanData()
print(X_data)
print(X_data.shape)
X_data = torch.tensor(X_data.values).float().to(device)



#print(X_data)

'''
class LinearRegression(nn.Module):
    def __init__(self, input_shape:int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Linear(in_features= input_shape,
                                                   out_features= hidden_units),
                                         nn.Linear(in_features= hidden_units,
                                                   out_features=hidden_units),
                                         nn.Linear(in_features=hidden_units,
                                                   out_features= output_shape),
                                         nn.ReLU())
    def forward(self, x):
        return self.layer_stack(x)
    
model = LinearRegression(input_shape=315, hidden_units=211, output_shape=1).to(device)
MODEL_SAVE_PATH = "model/first_model.pth"
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

model.eval()

with torch.inference_mode():
    preds = model(X_data)

print(preds)'''

#correct = torch.eq(y_data, preds).sum().item()
#acc = (correct / len(preds)) * 100
#
#print(f"Accuracy: {acc:.2f}%\n")