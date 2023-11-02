import numpy as np
import torch
import torch.nn as nn

criterion = torch.nn.BCEWithLogitsLoss()


y_hat = np.array([0.7, 0.3])

y_true = 1
y_true = torch.tensor(y_true).view(-1)
y_true = y_true.float()
y_hat = torch.tensor(y_hat)
y_hat = y_hat.float()

loss = - (y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

y_hat = torch.tensor([0.7, 0.8])

y_true = torch.tensor([1.0, 1.0])

loss = criterion(y_hat, y_true)
