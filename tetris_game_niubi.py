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





l = [[1,1,1,1,0],[1,0,1,1,1],[1,0,1,1,1],[0,0,0,0,0]]
camera=gamebox.Camera(1500, 1500)
game = Game(22, 10)
counter = 0

file = open('data3.txt')
batch_size = 64
batch = 0
temp_x = None
temp_y = None
count = 0


def tick(keys):
    global game, counter
    line = file.readline()
    line = line[1:-2]
    line = '[' + line + "]"
    temp  = json.loads(line)
    print(temp)
    camera.clear('grey')
    grid = temp[0]
    #print(grid)
    camera.draw(gamebox.from_color(90,90,"red",1000,1))
    camera.draw(gamebox.from_color(90, 595, "red", 1000, 1))
    camera.draw(gamebox.from_color(90, 90, "red", 1, 1000))
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] == 1:
                camera.draw(gamebox.from_color(100+x*21, 100+y*21, "black", 20, 20))
    camera.display()


ticks_per_second = 30
# keep this line the last one in your program
gamebox.timer_loop(ticks_per_second, tick)