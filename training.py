import torch
import random
from torch import nn
from core import Game
import torch.nn.functional as F
import matplotlib.pyplot as plt


WIDTH, HEIGHT = 12, 8
MAXIUM_ACTION_PER_FRAME = 5
MUTATION_RATE = 0.05
MUTATION_VALUE = 0.4
SEED = 2


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(WIDTH * HEIGHT, 10)
        self.linear2 = nn.Linear(10, 2)

        # Softmax operator.
        # This is log(exp(a_i) / sum(a))
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.id = random.randint(0,100000)

    def forward(self, x):
        gx = F.relu(self.linear(x))
        cx = F.relu(self.linear2(gx))
        yhat = self.log_softmax(cx)
        return yhat


def train(classifier):
    game = Game(WIDTH, HEIGHT, SEED)
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
            nextAction += 1 ## remove this
            if nextAction == 0:
                maxium_actions = MAXIUM_ACTION_PER_FRAME
            else:
                maxium_actions -= 1
def check_model(model):
    for p in model.parameters():
        print(p)

def mutate(dad, mom):
    child = Classifier()

    for (p,q,i) in zip(dad.parameters(), mom.parameters(), child.parameters()):
        if random.random() < 0.5:
            i.data = p.data.clone()
        else:
            i.data = q.data.clone()
    for p in child.parameters():
        for i in range(len(p)):
            if random.random() < MUTATION_RATE:
                temp =  0.1 * (random.random()-0.5)
                p.data[i] += temp
            if random.random() < 0.1:
                p.data[i] *= -1
    return child

temp = []
number_of_nn = 100
generation = 12
classifier = []
result = []
number_of_fittest = 10
for p in range(number_of_nn):
    classifier.append(Classifier())
for i in range(generation):
    for j in range(number_of_nn):
        if j % 10 == 0:
            print((j / number_of_nn)*100, "% done")
        r = train(classifier[j])
        temp.append((r, classifier[j]))
    temp.sort(key=lambda x : x[0])
    random.seed()
    selected_classifiers = temp[number_of_nn-number_of_fittest:number_of_nn] #Choose the best 10
    classifier[-1] = selected_classifiers[number_of_fittest-1][1] #keep the best classifier
    classifier[-2] = selected_classifiers[number_of_fittest - 2][1]  # keep the best classifier
    classifier[-3] = selected_classifiers[number_of_fittest - 3][1]  # keep the best classifier
    classifier[-4] = selected_classifiers[number_of_fittest - 4][1]  # keep the best classifier
    classifier[-5] = selected_classifiers[number_of_fittest - 5][1]  # keep the best classifier
    for k in range(number_of_nn-5):
        dad,mom = random.sample(selected_classifiers, 2)
        classifier[k] = mutate(dad[1], mom[1])
    temp.clear()
    print("-------------Generation Report "+str(i)+"---------------")
    print(((i+1)*100/generation), "% done")
    temp_list = [p[0] for p in selected_classifiers]
    id_list = [p[1].id for p in selected_classifiers]
    print("average fitness:", sum(temp_list)/number_of_fittest)
    print(temp_list)
    print(id_list)

