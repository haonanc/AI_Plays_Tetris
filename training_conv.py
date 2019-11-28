import torch
import random
from torch import nn
from core import Game
import torch.nn.functional as F


WIDTH, HEIGHT = 12, 12
MAXIUM_ACTION_PER_FRAME = 5


class Classifier(nn.Module):

    def __init__(self, geneID):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1,1,3, padding=1)
        self.linear = nn.Linear(36, 12)
        self.linear2 = nn.Linear(12, 4)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.geneID = geneID

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        qx = out.view(out.size(0), -1)
        gx = F.relu(self.linear(qx))
        cx = F.relu(self.linear2(gx))
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
            grid = torch.FloatTensor(returnValue).unsqueeze(0).unsqueeze(0)
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
    newClassifier = Classifier(classifier.geneID)
    for p in classifier.parameters():
        p.data -= p.data*(random.randint(-6, 6)/100)
    newClassifier.load_state_dict(classifier.state_dict())
    return newClassifier

temp = []
number_of_nn = 100
number_of_selected_nn = 20
rounds = 100
classifier = []
result = []
for p in range(number_of_nn):
    classifier.append(Classifier(str(p)))
for i in range(rounds):
    for j in range(number_of_nn):
        r = train(classifier[j])
        temp.append((r, classifier[j].geneID, classifier[j]))
    temp.sort(key=lambda x : x[0])
    selected_classifiers = temp[number_of_nn - number_of_selected_nn:number_of_nn] #Choose the best 5
    print("Evaluation Phase done")
    for k in range(number_of_nn - number_of_selected_nn):
        classifier[k] = mutate(selected_classifiers[k%len(selected_classifiers)][2])

    for k in range(number_of_nn - number_of_selected_nn, number_of_nn):
        temp_classifier = selected_classifiers[k % len(selected_classifiers)][2]
        newClassifier = Classifier(temp_classifier.geneID)
        newClassifier.load_state_dict(temp_classifier.state_dict())
        classifier[k] = temp_classifier
    print([(p[0],p[1]) for p in temp])
    temp.clear()
    print(((i+1)*100/rounds), "% done")
    print("--------------------------")