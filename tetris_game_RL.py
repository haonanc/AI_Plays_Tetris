import gamebox
import pygame

import torch
import random
from torch import nn
from core_new_niubi import Game
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json








l = [[1,1,1,1,0],[1,0,1,1,1],[1,0,1,1,1],[0,0,0,0,0]]
camera=gamebox.Camera(750, 1500)
game = Game(24, 10)
counter = 0
def tick(keys):
    global game, counter

    camera.clear('grey')
    grid = game.getRender_()
    camera.draw(gamebox.from_color(90,90,"black",2000,3))
    camera.draw(gamebox.from_color(90, 595, "black", 2000, 3))
    camera.draw(gamebox.from_color(301, 90, "black", 3, 2000))
    camera.draw(gamebox.from_text(470, 230, "AI FINAL PROJECT", "arial", 25, "black"))
    camera.draw(gamebox.from_text(450, 270, "Model:", "arial", 20, "black"))
    camera.draw(gamebox.from_text(450, 290, "Reinforce Learning", "arial", 20, "black"))
    camera.draw(gamebox.from_text(450, 330, "Score:", "arial", 20, "red"))
    camera.draw(gamebox.from_text(450, 350, str(game.score), "arial", 18, "red"))
    camera.draw(gamebox.from_color(90, 90, "black", 3, 2000))
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] == 1:
                camera.draw(gamebox.from_color(100+x*21, 100+y*21, "black", 20, 20))
    camera.display()

    if counter % 10 == 0:
        if not game.wrapper():
            return
    else:
        game.nextStep()
    counter += 1


ticks_per_second = 100
# keep this line the last one in your program
gamebox.timer_loop(ticks_per_second, tick)