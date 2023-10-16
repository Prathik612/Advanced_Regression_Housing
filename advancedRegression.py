import torch
from torch import nn
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}\n")

train_data = pd.read_csv("house-prices-advanced-regression-techniques\\train.csv")

#print(train_data.head)

#print(train_data.isnull().sum().to_string())
#print(train_data.shape)

#plt.figure(figsize = (8,6))
#sb.heatmap(train_data.isnull(), cbar=False , cmap = 'magma')
#plt.show()

def cleanData(data = train_data):
    data = data.drop(columns=["Id","Alley", "PoolQC", "Fence", "MiscFeature", "SalePrice"], errors='ignore', axis=1)

    num_cols = data.select_dtypes(np.number).columns
    cat_cols = data.select_dtypes(include=['object']).columns

    data_cat = pd.get_dummies(data[cat_cols], dummy_na=True)
    data_num = data[num_cols].apply(lambda x: (x - x.mean()) / (x.std()))
    #data_num = data_num.fillna(0)
    mean = data_num.mean(axis=0)
    data_num = data_num.fillna(mean)

    data = pd.concat([data_num, data_cat], axis=1)

    return data

y_data = train_data["SalePrice"].copy()
X_data = cleanData()
y_data = np.log(y_data, where=y_data != 0)
#print(X_data)
print(X_data.iloc[0].to_string())
print(X_data.shape)


#print(y_data.shape)

#X_data = torch.tensor(X_data.values).float()
#y_data = torch.tensor(y_data.values).float().unsqueeze(dim=1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, random_state=42)

#print(X_train.shape)
#print(y_train.shape)

X_train = torch.tensor(X_train.values).float().to(device)
y_train = torch.tensor(y_train.values).float().unsqueeze(dim=1).to(device)
X_test = torch.tensor(X_test.values).float().to(device)
y_test = torch.tensor(y_test.values).float().unsqueeze(dim=1).to(device)

BATCH_SIZE = 32
torch.manual_seed(42)

dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

class LinearRegressionv0(nn.Module):
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

class LinearRegressionv1(nn.Module):
    def __init__(self, input_shape:int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Linear(in_features= input_shape,
                                                   out_features= hidden_units),
                                         nn.ReLU(),
                                         nn.Linear(in_features= hidden_units,
                                                   out_features=hidden_units),
                                         nn.ReLU(),
                                         nn.Linear(in_features=hidden_units,
                                                   out_features= output_shape),)
    def forward(self, x):
        return self.layer_stack(x)
    
class LinearRegressionv2(nn.Module):
    def __init__(self, input_shape:int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Linear(in_features= input_shape,
                                                   out_features= hidden_units),
                                         nn.Linear(in_features= hidden_units,
                                                   out_features=hidden_units),
                                         nn.Linear(in_features= hidden_units,
                                                   out_features=hidden_units),
                                         nn.Linear(in_features=hidden_units,
                                                   out_features= output_shape),
                                         nn.ReLU())
    def forward(self, x):
        return self.layer_stack(x)
    
model = LinearRegressionv1(input_shape=314, hidden_units=211, output_shape=1).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.001)
epochs = 300

def train_step(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):
    
    train_loss = 0
    model.to(device)

    for batch, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(data_loader)
    
    return train_loss

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              device: torch.device = device):
    
    test_loss = 0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X ,y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)

        test_loss /= len(data_loader)
        return test_loss

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.dataloader, 
               loss_fn: torch.nn.Module,
               device: torch.device = device):
    loss = 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
        
        loss /= len(data_loader)
        loss = f"{loss:.3f}"

    return {"model_name": model.__class__.__name__,
            "model_loss": loss}

training_loss = []
testing_loss = []
epoch_count = []

for epoch in range(epochs):
    train_loss = train_step(data_loader=train_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        optimizer=optimizer)
    test_loss = test_step(data_loader=test_dataloader,
        model=model,
        loss_fn=loss_fn)
    
    if epoch%10 == 0:
        epoch_count.append(epoch)
        training_loss.append(train_loss.detach().cpu().numpy())
        testing_loss.append(test_loss.detach().cpu().numpy())
    
    if epoch%50 == 0:
        print(f"Epoch: {epoch}\n---------")
        print(f"Train loss: {train_loss:.5f}")
        print(f"Test loss: {test_loss:.5f}\n")


model_results = eval_model(model=model,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn)


test_data = pd.read_csv("house-prices-advanced-regression-techniques\\test.csv")

sub_id = test_data["Id"].copy()
X_sub = cleanData(test_data)
train_cols = X_data.columns
sub_cols = X_sub.columns
diff = train_cols.difference(sub_cols)

added_rows = 0 
idx = len(test_data) # Where added row should be modified

test_data_temp = test_data.copy() # Copy so we don't ruin the dataframe
temp_row = test_data.iloc[0] # Example row

for string in diff:
    test_data_temp.append(temp_row) # Append the example row at the end of dataframe
    col, value = string.split('_') # Split into column and value
    test_data_temp.loc[idx, col] = value
    idx += 1
    added_rows += 1

test_data_temp = cleanData(test_data_temp) # Preprocess the copy
X_sub = test_data_temp.drop(test_data_temp.tail(added_rows).index) # Remove the added rows
X_sub = torch.tensor(X_sub.values).float().to(device)

y_sub = model(X_sub).squeeze() # Make it into array
y_sub = torch.exp(y_sub) # Scale back to normal value 
y_sub = y_sub.cpu().detach().numpy() # Change to numpy
submission = pd.DataFrame({"Id":sub_id, "SalePrice":y_sub}) # Make dataframe for submissino
submission.to_csv('submission.csv', index=False)


#print("New: \n")
#print(train_data.isnull().sum().to_string())
#print(X_data.shape)
#print(y_data.shape)