import torch
import random
from torch import nn
from core_new import Game
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

WIDTH, HEIGHT = 24, 10
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

temp = []
classifier = Classifier()
classifier.load_state_dict(torch.load('model2.pkl'))
for i in range(10000):
    game = Game(24,10)
    if(i %100 == 0):
        print(i)
    while True:

        grid = game.getRender_new()
        grid[0].append(0)
        grid[0].append(0)
        grid[0].append(0)
        processed_grid = torch.FloatTensor(grid).reshape(-1)
        output = classifier(processed_grid)
        output2 = torch.argsort(output, descending=True)
        if not game.wrapper(output2):
            break
    temp.append(game.score)

print(max(temp))


