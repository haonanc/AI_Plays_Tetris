import torch
from torch import nn
from core import Game
import matplotlib.pyplot as plt


WIDTH, HEIGHT = 24, 10
MAXIUM_ACTION_PER_FRAME = 5

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(WIDTH * HEIGHT, 12)
        self.linear2 = nn.Linear(12, 4)

        # Softmax operator.
        # This is log(exp(a_i) / sum(a))
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        gx = self.linear(x)
        cx = self.linear2(gx)
        yhat = self.log_softmax(cx)
        return yhat


def train():
    classifier = Classifier()
    game = Game(WIDTH, HEIGHT)
    nextAction = 0
    maxium_actions = MAXIUM_ACTION_PER_FRAME
    while True:
        alive, returnValue = game.wrapper(nextAction)
        if not alive:
            return returnValue
        else:
            grid = torch.FloatTensor(returnValue).reshape(-1)
        if maxium_actions == 0:
            maxium_actions = MAXIUM_ACTION_PER_FRAME
            nextAction = 0
        else:
            output = classifier(grid)
            _, nextAction = output.max(0)
            if nextAction == 0:
                maxium_actions = MAXIUM_ACTION_PER_FRAME
            else:
                maxium_actions -= 1
temp = []
number_of_nn = 100
for i in range(number_of_nn):
    r = train()
    temp.append(r)
    if i % (number_of_nn / 10) == 0:
        print(i / (number_of_nn / 100), "% done")
print(sorted(temp))
