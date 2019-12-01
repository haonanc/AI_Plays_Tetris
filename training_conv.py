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
    # Convolutional layers.

    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, 2)

        # Linear layers.
        self.fc1 = nn.Linear(48, 240)
        self.fc2 = nn.Linear(240, 84)
        self.fc3 = nn.Linear(240, 37)

    def forward(self, x):
        # Conv1 + ReLU + MaxPooling.
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)

        # Conv2 + ReLU + MaPooling.
        # out = F.relu(self.conv2(out))
        # out = F.max_pool2d(out, 2)

        # This flattens the output of the previous layer into a vector.
        out = out.view(out.size(0), -1)
        # Linear layer + ReLU.
        out = F.relu(self.fc1(out))
        # Another linear layer + ReLU.
        # out = F.relu(self.fc2(out))
        # A final linear layer at the end.
        out = self.fc3(out)

        # We will not add nn.LogSoftmax here because nn.CrossEntropy has it.
        # Read the documentation for nn.CrossEntropy.
        return out

dataLoader = []
file = open('data2.txt')
batch_size = 128
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
        temp_x = torch.FloatTensor(temp[0]).unsqueeze(0).unsqueeze(0)
        temp_y = torch.LongTensor([temp[1]])
    else:
        new_x = torch.FloatTensor(temp[0]).unsqueeze(0).unsqueeze(0)
        new_y = torch.LongTensor([temp[1]])
        temp_x = torch.cat((temp_x,new_x),0)
        temp_y = torch.cat((temp_y,new_y),0)
    batch += 1

print(len(dataLoader))
#print(dataLoader[0][0])
#print(dataLoader[0][1])
random.shuffle(dataLoader)
train_len = 128 * 50
val_len = 8281 - 128 * 50
trainLoader = dataLoader[:50]
valLoader = dataLoader[50:-1]

# Re-initialize the classifier.
classifier = Classifier()

# Let's define the loss function or criterion.
criterion = nn.CrossEntropyLoss()

# Number of epochs is the number of times we go over the full training set.
num_epochs = 150

# Learning rate.
learningRate = 0.01
classifier.train()
# We could also use the following pytorch utility. It supports momentum!
# optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.001, momentum = 0.9)

# This is often needed to prepare model for training.
# But it does almost nothing, it doesn't train the model.

# Short-cut for the model parameters.

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

torch.save(classifier.state_dict(), "model.pkl")