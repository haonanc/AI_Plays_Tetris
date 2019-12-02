import gamebox
import core_new
import pygame

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
#
# temp = []
classifier = Classifier()
classifier.load_state_dict(torch.load('model2.pkl'))
classifier.eval()
# for i in range(10000):
#     game = Game(24,10)
#     if(i %100 == 0):
#         print(i)
#     while True:
#
#         grid = game.getRender_new()
#         grid[0].append(0)
#         grid[0].append(0)
#         grid[0].append(0)
#         processed_grid = torch.FloatTensor(grid).reshape(-1)
#         output = classifier(processed_grid)
#         output2 = torch.argsort(output, descending=True)
#         if not game.wrapper(output2):
#             break
#     temp.append(game.score)

# print(max(temp))





l = [[1,1,1,1,0],[1,0,1,1,1],[1,0,1,1,1],[0,0,0,0,0]]
camera=gamebox.Camera(750, 1500)
game = Game(24, 10)
counter = 0
def tick(keys):
    global game, counter

    camera.clear('grey')
    grid = game.getRender_()
    #print(grid)
    camera.draw(gamebox.from_color(90,90,"black",2000,3))
    camera.draw(gamebox.from_color(90, 595, "black", 2000, 3))
    camera.draw(gamebox.from_color(301, 90, "black", 3, 2000))
    camera.draw(gamebox.from_text(470, 230, "AI FINAL PROJECT", "arial", 25, "black"))
    camera.draw(gamebox.from_text(450, 270, "Model:", "arial", 20, "black"))
    camera.draw(gamebox.from_text(450, 290, "Neural Network", "arial", 20, "black"))
    camera.draw(gamebox.from_text(450, 330, "Score:", "arial", 20, "red"))
    camera.draw(gamebox.from_text(450, 350, str(game.score), "arial", 18, "red"))
    camera.draw(gamebox.from_color(90, 90, "black", 3, 2000))

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] == 1:
                camera.draw(gamebox.from_color(100+x*21, 100+y*21, "black", 20, 20))
    camera.display()

    if counter % 30 == 0:
        grid = game.getRender_new()
        grid[0].append(0)
        grid[0].append(0)
        grid[0].append(0)
        processed_grid = torch.FloatTensor(grid).reshape(-1)
        output = classifier(processed_grid)
        output2 = torch.argsort(output, descending=True)
        if not game.wrapper(output2):
            return
    else:
        game.nextStep()
    counter += 1


ticks_per_second = 30
# keep this line the last one in your program
gamebox.timer_loop(ticks_per_second, tick)