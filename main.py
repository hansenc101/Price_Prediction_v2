# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:58:52 2023

@author: Christopher Hansen
email: hansenc101@gmail.com
"""

import torch
import LightningLearning as LL
import TorchUtils as tu
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import TQDMProgressBar as bar
#import wandb
#from pytorch_lightning.loggers import WandbLogger


#%% Retrieve Stock Data 
predict_stock='msft'
feature_stocks=[predict_stock, 'tsla','meta','sony','amzn','nflx','gbtc','gdx','intc','dal',
                'c','goog','aapl','msft','ibm','hp','orcl','sap','crm','hubs','twlo']
n_features = len(feature_stocks)-1
x_file_path = 'x_data.csv'
y_file_path = 'y_data_next_day.csv'
y_same_file_path = 'y_data_same_day.csv'

X = np.genfromtxt(x_file_path, delimiter=',', dtype=float)
X = X.astype(np.float32)
Y = np.genfromtxt(y_file_path, delimiter=',', dtype=float) # use this for next day prediction training
#Y = np.genfromtxt(y_same_file_path, delimiter=',', dtype=float) # use this for same day prediction training
Y = Y.astype(np.float32)

#% Scale the input data using min/max scaling, by each feature
#X, x_min, x_max = tu.min_max_scale(X)
Y, y_min, y_max = tu.min_max_scale(Y, axis=0) # scale the target data
X, x_min, x_max = tu.min_max_scale(X, axis=1) # scale the input data
#%% Dataset processing and create dataloaders
days = 45
stockData = tu.StockDataset(X,Y,days=days)

# set batch_size to 0 to use entire dataset as batch
batch_size = 8
stockDataModule = tu.StockDataModule(stockData, batch_size=batch_size) 
train_dataloader = stockDataModule.train_dataloader()
val_dataloader = stockDataModule.val_dataloader()
test_dataloader = stockDataModule.test_dataloader()

#%% Create the model
# configuration for the model
class Config(): # I should change this to a dictionary instead of a class
    def __init__(self, n_input_size=1, hidden_size=1, n_layers=1, lr=1e-4, dropout=0, y_min=0, y_max=1):
        self.learning_rate = lr
        self.dropout = dropout
        self.n_input_size = n_input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

input_size = (days, batch_size, n_features) # input size: (n_samples, batch_size, n_features/sample)
config = Config(n_input_size=input_size, hidden_size=160, n_layers=12, lr=1e-5, dropout=0.1)

# Create the model 
lightning_model = LL.StockLightningModule(config=config)

#%% Prepare system - determine if using gpu or cpu
used_gpu = False
num_epochs = 25
if torch.cuda.is_available():
  print('\n--Using GPU for training--\n')
  torch.set_default_tensor_type(torch.FloatTensor)
  torch.backends.cudnn.benchmark = True
  torch.set_float32_matmul_precision('high')
  trainer = pl.Trainer(accelerator='gpu', max_epochs=num_epochs, 
                       callbacks=[bar(refresh_rate=1)])
else:
  print('\n--Using CPU for training--\n')
  trainer = pl.Trainer(max_epochs=num_epochs, 
                       callbacks=[bar(refresh_rate=1)])

#%% Train the model
trainer.fit(lightning_model, train_dataloader, val_dataloader)

#%% Gather loss data
train_data = lightning_model.get_train_loss_data()
val_data = lightning_model.get_val_loss_data()
val_pred = tu.min_max_undo(lightning_model.get_val_predictions(), max=y_max, min=y_min)
val_target = tu.min_max_undo(lightning_model.get_val_targets(), max=y_max, min=y_min)


# get loss data off of gpu to cpu and convert to np.arrays
train_data = np.array([tensor.cpu().detach().numpy() for tensor in train_data])
val_data = np.array([tensor.cpu().detach().numpy() for tensor in val_data])
min_train = np.min(train_data) 
min_val = np.min(val_data) 
final_train = train_data[-1]
final_val = val_data[-1]

print('')
print('')
print('============== Unscaled Raw Losses ================')
print('Minimum Training MSE Loss:   ', min_train)
print('Minimum Validation MSE Loss: ', min_val)
print('Final Training MSE Loss:     ', final_train)
print('Final Validation MSE Loss:   ', final_val)
print('')

# Scale the losses back to the original target data scale
min_train = tu.min_max_undo(min_train, y_min, y_max)
min_val = tu.min_max_undo(min_val, y_min, y_max)
final_train = tu.min_max_undo(final_train, y_min, y_max)
final_val = tu.min_max_undo(final_val, y_min, y_max)

print('=== Scaled Losses (Original Target Data Scale) ====')
print('Minimum Training MSE Loss:   ', min_train)
print('Minimum Validation MSE Loss: ', min_val)
print('Final Training MSE Loss:     ', final_train)
print('Final Validation MSE Loss:   ', final_val)
print('\n')

#% Plot the training and validation loss curves
train_steps = np.linspace(0, len(train_data), len(train_data))
val_steps = np.linspace(0, len(train_data), len(val_data))
plt.plot(train_steps, train_data, label='Training Loss')
plt.plot(val_steps, val_data, label='Validation Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Testing and Validation Loss using Lightning')
plt.legend()
plt.show()

#%% Plot Validation Predictions vs Actual Stock Prices
n_days = 30
preds = val_pred[-1*n_days:-1]
actual = val_target[-1*n_days:-1]
Days = np.linspace(1, len(val_pred[-1*n_days:-1]), len(val_pred[-1*n_days:-1]))
plt.plot(Days, actual, label='Actual Stock Price')
plt.plot(Days, preds, label='Predicted Stock Price')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Actual and Predicted Stock Prices')
plt.legend()
plt.show()

accuracy = 100 - ((abs(preds-actual)/actual)*100)
mean = np.mean(accuracy)
print(f'The average accuracy of a prediction is:  {mean:.2f}%')
print(f'The maximum accuracy of a prediction was: {np.max(accuracy):.2f}%')
print(f'The minimum accuracy of a prediction was: {np.min(accuracy):.2f}%')
# Plot the prediction accuracy curve
plt.plot(Days, accuracy, label='Accuracy')
plt.axhline(y=99, color='g', linestyle='--')
plt.xlabel('Days')
plt.ylabel('Prediction Accuracy')
plt.title('Percentage Difference between Actual and Predicted Prices')
plt.legend()
plt.show()

#%% Save the model
ans = input('Do you wish to save the model? (y/n) \n')
ans = ans.lower()
save = False
if (ans=='y') or (ans=='yes'): 
    save = True
    print('Saving model...')
    lightning_model.save()
else:
    print('Model will not be saved...\n')
input('Press [Enter] to finish...')
