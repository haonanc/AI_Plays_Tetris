import torch
import random
from torch import nn
from core import Game
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json


WIDTH, HEIGHT = 24, 10
SEED = 2


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(WIDTH * HEIGHT + 10, 1600)
        self.linear2 = nn.Linear(1600,480)
        self.linear5 = nn.Linear(480, 37)
        self.dropout = torch.nn.Dropout(0.1)

        # Softmax operator.
        # This is log(exp(a_i) / sum(a))
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.id = random.randint(0,100000)

    def forward(self, x):
        x = F.leaky_relu(self.linear(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        # x = self.dropout(x)
        # x = F.leaky_relu(self.linear3(x))
        # x = self.dropout(x)
        # x = F.leaky_relu(self.linear4(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear5(x))
        return x
        yhat = self.log_softmax(x)
        return yhat

dataLoader = []
file = open('data3.txt')
batch_size = 64
batch = 0
temp_x = None
temp_y = None
count = 0
for line in file:
    count += 1
    line = line[1:-2]
    line = '[' + line + "]"
    temp  = json.loads(line)
    temp[0][0].append(0)
    temp[0][0].append(0)
    temp[0][0].append(0)
    if batch % batch_size == 0:
        if batch != 0:
            dataLoader.append((temp_x,temp_y))
        temp_x = torch.FloatTensor(temp[0]).reshape(-1).unsqueeze(0)
        temp_y = torch.LongTensor([temp[1]])
    else:
        new_x = torch.FloatTensor(temp[0]).reshape(-1).unsqueeze(0)
        new_y = torch.LongTensor([temp[1]])
        temp_x = torch.cat((temp_x,new_x),0)
        temp_y = torch.cat((temp_y,new_y),0)
    batch += 1

train_batch = int((count/batch_size) *.9)
print(len(dataLoader))
#print(dataLoader[0][0])
#print(dataLoader[0][1])
random.shuffle(dataLoader)
train_len = train_batch * batch_size
val_len = count - train_batch * batch_size
trainLoader = dataLoader[:train_batch]
valLoader = dataLoader[train_batch:-1]

# Re-initialize the classifier.
classifier = Classifier()

# Let's define the loss function or criterion.
criterion = nn.CrossEntropyLoss()

# Number of epochs is the number of times we go over the full training set.
num_epochs = 100

# Learning rate.
learningRate = 0.01
classifier.train()
# We could also use the following pytorch utility. It supports momentum!
# optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.001, momentum = 0.9)

# This is often needed to prepare model for training.
# But it does almost nothing, it doesn't train the model.

# Short-cut for the model parameters.
weight = classifier.linear.weight
bias = classifier.linear.bias
#classifier.load_state_dict(torch.load('model.pkl'))
optimizer = optim.SGD(classifier.parameters(), lr = learningRate, momentum=0.9)
# Training loop starts.
for epoch in range(0, num_epochs):

    correct = 0
    cumloss = 0

    # Go over all the training data one batch at a time.
    for (i, (x, y)) in enumerate(trainLoader):
        # Flatten the images in the batch to input to the classifier.

        # Compute predictions under the current model.
        yhat = classifier(x)

        # Compute the loss with respect to this batch.
        loss = criterion(yhat, y)

        # Set to zero gradients computed in previous iteration.
        optimizer.zero_grad()
        # optimizer.zero_grad()

        # Compute the gradients for the entire model variables,
        # this includes inputs, outputs, and parameters (weight, and bias).
        loss.backward()

        # Now we can update the weight and bias parameters.
        optimizer.step() # We can also use this utility from pytorch.

        with torch.no_grad():  # Stop recording operations for gradient computation.
            # Count the number of correct predictions on this batch.
            _, max_labels = yhat.max(1)  # Extract the max values from yhat.
            # Check if those match the correct values in y.
            # and store in cumulative variable of correct values so far during training.
            correct += (max_labels == y).sum().item()
            # Also compute the cumulative loss so far during training.
            cumloss += loss.item()

    n = train_len
    print('({0}) Training: loss = {1:.4f}, accuracy = {2:.4f}'
          .format(epoch, cumloss / n, correct / n))

    correct = 0
    cumloss = 0
    # Compute accuracy on the test set.
    with torch.no_grad():  # Do not record operations for grad computation.
        for (i, (x, y)) in enumerate(valLoader):
            # Flatten the images in the batch to input to the classifier.

            # Compute predictions under the current model.
            yhat = classifier(x)

            # Check if those match the correct values in y.
            # and store in cumulative variable of correct values so far.
            _, max_labels = yhat.max(1)
            #print(y,max_labels)
            correct += (max_labels == y).sum().item()
            # Also compute the cumulative loss so far.
            cumloss += loss.item()

    n = val_len
    print('({0}) Validation: loss = {1:.4f}, accuracy = {2:.4f}'
          .format(epoch, cumloss / n, correct / n))
    print('\n')

torch.save(classifier.state_dict(), "model3.pkl")