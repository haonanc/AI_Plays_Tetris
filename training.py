import torch
import random
from torch import nn
from core import Game
import matplotlib.pyplot as plt


WIDTH, HEIGHT = 24, 10
MAXIUM_ACTION_PER_FRAME = 5
MUTATION_RATE = 0.05
MUTATION_VALUE = 0.2


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

def mutate(dad, mom):
    child = Classifier()
    for (p,q,i) in zip(dad.parameters(), mom.parameters(), child.parameters()):
        if random.random() < 0.5:
            i.data = p.data.clone()
        else:
            i.data = q.data.clone()

    for p in child.parameters():
        if random.random() < MUTATION_RATE:
            p.data = p.data + MUTATION_VALUE*random.random()*2 - MUTATION_VALUE
    return child

temp = []
number_of_nn = 100
generation = 100
classifier = []
result = []
number_of_fittest = 5
for p in range(number_of_nn):
    classifier.append(Classifier())
for i in range(generation):
    for j in range(number_of_nn):
        r = train(classifier[j])
        temp.append((r, classifier[j]))
    temp.sort(key=lambda x : x[0])
    selected_classifiers = temp[number_of_nn-number_of_fittest:number_of_nn] #Choose the best 5
    classifier[number_of_nn-1] = selected_classifiers[number_of_fittest-1][1] #keep the best classifier
    for k in range(number_of_nn-1):
        dad,mom = random.sample(selected_classifiers, 2)
        classifier[k] = mutate(dad[1], mom[1])
    temp.clear()
    print([p[0] for p in selected_classifiers])
    print(((i+1)*100/generation), "% done")
    print("-------------------------------------")