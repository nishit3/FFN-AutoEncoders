import sys
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn as nn


data = pd.read_csv('mnist_train.csv', sep=',')
data = data[data.keys().drop('label')]

dataT = torch.tensor(data.to_numpy()).float()
dataT = torch.div(dataT, 255.00)

batch_size = 64
n_epochs = 3


class MnistAE(nn.Module):
    def __init__(self, EDUnits, latentUnits):
        super().__init__()
        self.inputL = nn.Linear(len(data.keys()), EDUnits)
        self.encL = nn.Linear(EDUnits, latentUnits)
        self.latentL = nn.Linear(latentUnits, EDUnits)
        self.decL = nn.Linear(EDUnits, len(data.keys()))

    def forward(self, X):
        X = nn.functional.relu(self.inputL(X))
        X = nn.functional.relu(self.encL(X))
        X = nn.functional.relu(self.latentL(X))
        return nn.functional.sigmoid(self.decL(X))


EDunits_list = np.linspace(start=10, stop=501, num=12, dtype=int)
latentUnits_list = np.linspace(start=5, stop=100, num=8, dtype=int)
results = np.zeros(shape=(len(EDunits_list), len(latentUnits_list)))
total_iterations = len(EDunits_list)*len(latentUnits_list)
start_time = time.process_time()

for i, EDunit in enumerate(EDunits_list):
    for j, latentUnit in enumerate(latentUnits_list):
        AutoEncoder = MnistAE(EDUnits=EDunit, latentUnits=latentUnit)
        optimizer = torch.optim.Adam(AutoEncoder.parameters(), lr=.01)
        lossFun = nn.MSELoss()
        final_loss = 0.00

        for epoch_i in range(n_epochs):
            x = 0
            while x < len(data):
                train_set = dataT[x:x + batch_size]
                preds = AutoEncoder(train_set)
                loss = lossFun(train_set, preds)
                if epoch_i == n_epochs - 1 and x + 3 * batch_size >= len(data):
                    final_loss = final_loss + loss.item() / 3.00
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sys.stdout.flush()
                x = x + batch_size
        results[i, j] = final_loss
        msg = f"Encoder/Decoder Units = {EDunit}    Latent Layer Units = {latentUnit}   Loss = {final_loss}"
        sys.stdout.write('\r'+msg)

end_time = time.process_time()
print(f'\n\nTotal Time = {(end_time - start_time)/60.00} mins')

plt.imshow(results, cmap='Purples')
plt.colorbar()
plt.show()
