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

# https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

le = LabelEncoder()
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
    return np.mean(training_loss)

# make a class prediction for one row of data

def predict(row, model):
    # convert row to data
    row = torch.tensor([row], dtype=torch.float32).cuda()
    model.eval()
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().cpu().numpy()
    return np.argmax(yhat)
    

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
    f1_metric = f1_score(vstack(actuals), vstack(predictions), average = "macro")
    print('F1 score:' , f1_metric)
    return np.mean(valid_loss), np.mean(valid_acc)

def get_dataloaders(df, batch_size, valid_culture=None):
        
    ### For random splitting
    # if valid_culture is None:
    Y = df[['emotion']].values
    X = df.drop(['frame', 'face_id', 'culture', 'filename', 'emotion', 'confidence','success'], axis=1)
    Y = le.fit_transform(Y)
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    # lengths = [int(len(dataset)*0.7), len(dataset) - int(len(dataset)*0.7)]
    # train, valid = random_split(dataset, lengths)
    ### For cultural splitting
    # else:

    #     valid_df = df[df['culture'] == valid_culture]
    #     train_df = df[df['culture'] != valid_culture]

    #     train_labels = train_df[['emotion']].values
    #     valid_labels = valid_df[['emotion']].values
    #     train_labels = le.fit_transform(train_labels)
    #     valid_labels = le.fit_transform(valid_labels)

    #     valid_df.drop(['frame', 'face_id', 'culture', 'filename', 'emotion', 'confidence','success'], axis=1, inplace=True)
    #     train_df.drop(['frame', 'face_id', 'culture', 'filename', 'emotion', 'confidence','success'], axis=1, inplace=True)

    #     Y_tensor_train = torch.tensor(train_labels, dtype=torch.long)
    #     Y_tensor_valid = torch.tensor(valid_labels, dtype=torch.long)

    #     X_tensor_train = torch.tensor(train_df.values, dtype=torch.float32)
    #     X_tensor_valid = torch.tensor(valid_df.values, dtype=torch.float32)
    #     train = TensorDataset(X_tensor_train, Y_tensor_train)
    #     valid = TensorDataset(X_tensor_valid, Y_tensor_valid)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    # valid_dataloader = DataLoader(valid, shuffle=False, batch_size=batch_size)
    # test_dataloader = DataLoader(test,  shuffle=False, batch_size=batch_size)
    return dataloader


net = ContemptNet()
if torch.cuda.is_available():
    net.cuda()
# print(net)
batch_size = 32
epochs = 100
# ASGD worked ok. (layers = 1, units=32, F1=0.45, acc=0.51)
# optimizer = optim.ASGD(net.parameters(), lr=0.005)
optimizer = optim.SGD(net.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

data_path = './processed_data_csv/all_videos.csv'
df = pd.read_csv(data_path)
valid_files = pd.Series(df['filename'].unique()).sample(frac=0.3)
test_files = pd.Series(valid_files).sample(frac=0.15)
test_df = df[df['filename'].isin(test_files)]
valid_df = df[df['filename'].isin(valid_files)]
# test_df=df.sample(n=300,random_state=200)
df = df[~df['filename'].isin(valid_files)]
df = df[~df['filename'].isin(test_files)]
train_dataloader = get_dataloaders(df, batch_size)
valid_dataloader = get_dataloaders(valid_df, batch_size)

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
# plt.legend(loc="lower right")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()
# evaluate_model(test_dataloader, net)

Yhat = list()
# Y = le.fit_transform(test_df['emotion'].values)
test_df_copy = test_df.drop(['frame', 'face_id', 'culture', 'filename', 'emotion', 'confidence','success'], axis=1)
for row in test_df_copy.values:
    p = predict(row, net)
    # p = p.reshape((len(p), 1))
    Yhat.append(p)

# test_df['integer_emotion'] = Y
test_df['predicted'] = le.inverse_transform(Yhat)
print('Len test_df: ', len(test_df))
# print('Len Y:', len(Y))
print(test_df.sample(25))
print('Test accuracy: %.3f' % (accuracy_score(le.fit_transform(test_df['emotion'].values), Yhat)))
# actual = actual.reshape((len(actual), 1))
# yhat = yhat.reshape((len(yhat), 1))