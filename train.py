import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from numpy import vstack
from network import ContemptNet
import matplotlib.pyplot as plt
#https://www.kaggle.com/sirojiddin/titanic-mlp-pytorch



def train_model(train_dl, model):
    training_loss = []
    for i, data in enumerate(train_dataloader):
        # print()

        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        model = model.train()
        # forward + backward + optimize
        yhat = model(inputs)
        loss = criterion(yhat, labels)
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
        # print statistics
    return np.mean(training_loss)

# def test_model(test_dl, model):
#     predictions, actuals = list(), list()
#     for i, data in enumerate(test_dl):
#         inputs, targets = data
#         # evaluate the model on the test set
#         yhat = model(inputs)
#         # retrieve numpy array
#         yhat = yhat.detach().numpy()
#         actual = targets.numpy()
#         actual = actual.reshape((len(actual), 1))
#         # round to class values
#         yhat = yhat.reshape((len(yhat), 1))
#         # store
#         predictions.append(yhat)
#         actuals.append(actual)
#     predictions, actuals = vstack(predictions), vstack(actuals)
#     # calculate accuracy
#     return predictions, actuals

def valid_model(valid_dl, model):
    valid_loss = []
    valid_acc = []
    predictions, actuals = list(), list()
    model.eval()
    for i, data in enumerate(valid_dl):
        inputs, targets = data
        inputs = inputs.cuda()
        targets = targets.cuda()
        # retrieve numpy array
        yhat = model(inputs)
        # print("yhat shape: ", yhat.shape)
         # retrieve numpy array
        loss = criterion(yhat, targets)
        yhat = yhat.cpu().detach().numpy()
        actual = targets.cpu().numpy()
        # convert to class labels
        yhat = np.argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # calculate accuracy
        predictions.append(yhat)
        actuals.append(actual)
        acc = accuracy_score(vstack(predictions), vstack(actuals))
        # round to class values
        # yhat = yhat.reshape((len(yhat), 1))
        valid_loss.append(loss.item())
        valid_acc.append(acc)

    # f1 = f1_score(actuals, predictions)
    # print('F1 score:' , f1)
    return np.mean(valid_loss), np.mean(valid_acc)

net = ContemptNet()
if torch.cuda.is_available():
    net.cuda()
print(net)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

data_path = './processed_data_csv/all_videos.csv'
df = pd.read_csv(data_path)
Y = df[['emotion']].values
X = df.drop(['frame', 'face_id', 'culture', 'filename', 'emotion', 'confidence','success'], axis=1)
Y = LabelEncoder().fit_transform(Y)

Y_tensor = torch.tensor(Y, dtype=torch.long)
# Y_tensor = Y_tensor.reshape((len(Y_tensor), 1))
X_tensor = torch.tensor(X.values, dtype=torch.float32)
batch_size = 32
epochs = 64
dataset = TensorDataset(X_tensor, Y_tensor)
train, valid, test = random_split(dataset, [13024, 5800, 200 ])
train_dataloader = DataLoader(train, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(valid, shuffle=False, batch_size=batch_size)
test_dataloader = DataLoader(test,  shuffle=False, batch_size=batch_size)

train_losses = []
valid_losses = []
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)
    epoch_train_loss = train_model(train_dataloader, net)
    epoch_valid_loss, epoch_valid_acc = valid_model(valid_dataloader, net)
    train_losses.append(epoch_train_loss)
    valid_losses.append(epoch_valid_loss)
    print('[%d] Training loss: %.3f' %
                (epoch + 1, epoch_train_loss ))
    print('[%d] Validation loss: %.3f' %
                (epoch + 1, epoch_valid_loss ))
    print('[%d] Validation accuracy: %.3f' %
                (epoch + 1, epoch_valid_acc ))

plt.plot(train_losses, label='training loss')
plt.plot(valid_losses, label='validation loss')
plt.show()
# evaluate_model(test_dataloader, net)

# https://medium.com/analytics-vidhya/classifying-iris-dataset-using-pytorch-31ccd7716803
# def train_epoch(model, opt, criterion, batch_size=50):
#     model.train()
#     losses = []
#     for beg_i in range(0, X_train.size(0), batch_size):
#         x_batch = X_train[beg_i:beg_i + batch_size, :]
#         y_batch = y_train[beg_i:beg_i + batch_size, :]
#         x_batch = Variable(x_batch)
#         y_batch = Variable(y_batch)

#         opt.zero_grad()
#         # (1) Forward
#         y_hat = net(x_batch)
#         # (2) Compute diff
#         loss = criterion(y_hat, y_batch)
#         # (3) Compute gradients
#         loss.backward()
#         # (4) update weights
#         opt.step()        
#         losses.append(loss.data.numpy())
#     return losses