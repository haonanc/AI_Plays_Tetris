import torch
import random
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


def train(classifier):
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

def mutate(classifier):
    for p in classifier.parameters():
        p.data -= p.data*(random.randint(-25, 25)/100)
    return classifier

temp = []
number_of_nn = 100
rounds = 5
classifier = []
result = []
for p in range(number_of_nn):
    classifier.append(Classifier())
for i in range(rounds):
    for j in range(number_of_nn):
        r = train(classifier[j])
        temp.append((r, classifier[j]))
    temp.sort(key=lambda x : x[0])
    selected_classifiers = temp[number_of_nn-5:number_of_nn] #Choose the best 5
    for k in range(number_of_nn):
        classifier[k] = mutate(selected_classifiers[k%len(selected_classifiers)][1])
    temp.clear()
    print([p[0] for p in selected_classifiers])
    print(((i+1)*100/rounds), "% done")
    print("--------------------------")