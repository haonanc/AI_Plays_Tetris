from enum import Enum
import random
import copy

#<<Reward Functions>>
SURVIVAL_REWARD = 0
ACTION_REWARD = 0
TETRIS_REWARD = 1000


class Piece:

    def __init__(self):
        global GlOBAL_TEST
        self.orientation = 0
        self.shapes = []
        function = [self.O,self.Test]
        #function = [self.I, self.O, self.T, self.S, self.Z]
        function[random.randint(0, len(function) - 1)]()

    def Test(self):
        self.shapes = [[[1]]]

    def I(self):
        self.shapes = [[[1, 1, 1, 1]], [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]]

    def O(self):
        self.shapes = [[[1, 1], [1, 1]]]

    def TESTO(self):
        self.shapes = [[[1, 1]]]

    def T(self):
        self.shapes = [[[1, 1, 1], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 1]],
                       [[0, 0, 1], [0, 1, 1], [0, 0, 1]]]

    def S(self):
        self.shapes = [[[0, 1, 1], [1, 1, 0]], [[1, 0, 0], [1, 1, 0], [0, 1, 0]]]

    def Z(self):
        self.shapes = [[[1, 1, 0], [0, 1, 1]], [[1, 0, 0], [1, 1, 0], [0, 1, 0]]]

    def rotate(self):
        if self.orientation == len(self.shapes) - 1:
            self.orientation = 0
        else:
            self.orientation += 1

    def getShape(self):
        return self.shapes[self.orientation]

    def getNextShape(self):
        if self.orientation == len(self.shapes) - 1:
            return self.shapes[0]
        else:
            return self.shapes[self.orientation + 1]


class Action(Enum):
    down = 0
    left = 1
    right = 2
    rotate = 3

class Game:

    def __init__(self, width, height, seed=None):
        if seed:
            random.seed(seed)
        self.board = [[0 for _y in range(height)] for _x in range(width)]
        self.score = 0
        self.current_piece_location = [0, 0]  # [ycoordinate, xcoordinate]
        self.nextPiece()
        return

    def doAction(self, action): # when return false, game over
        checkRotate = False
        if action == 0:
            self.score += SURVIVAL_REWARD
            dy, dx = 1, 0
        elif action == 1:
            dy, dx = 0, -1
        elif action == 2:
            dy, dx = 0, 1
        elif action == 3:
            dy, dx = 0, 0
            checkRotate = True
        else:
            raise Exception("Illegeal Action Exception")
        if not self.checkObstruction(dy, dx, rotate=checkRotate):
            if action == 0 and not self.checkTetris():
                return False
            return True
        self.current_piece_location = [self.current_piece_location[0] + dy, self.current_piece_location[1] + dx]
        if action == 3:
            self.current_piece.rotate()
        return True

    def nextPiece(self):
        self.current_piece = Piece()
        self.current_piece_location = [0, 0]
        if self.checkObstruction(0, 0):
            return True
        return False

    def checkObstruction(self, dy, dx, rotate=False):
        if rotate:
            shape = self.current_piece.getNextShape()
        else:
            shape = self.current_piece.getShape()
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x] == 0:
                    continue
                newY = y + dy + self.current_piece_location[0]
                newX = x + dx + self.current_piece_location[1]
                if not (0 <= newY < len(self.board) and 0 <= newX < len(self.board[0]) and self.board[newY][newX] == 0):
                    return False
        return True

    def checkTetris(self):
        shape = self.current_piece.getShape()
        dy, dx = self.current_piece_location[0], self.current_piece_location[1]
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x] == 1:
                    self.board[dy + y][dx + x] = 1
        newBoard = [[0 for _x in range(len(self.board[0]))] for _y in range(len(self.board))]
        layer = len(self.board) - 1
        for i in range(len(self.board)-1,-1,-1):
            if sum(self.board[i]) == len(self.board[0]):
                self.score += TETRIS_REWARD
            else:
                newBoard[layer] = self.board[i]
                layer -= 1
        self.board = newBoard
        if not self.nextPiece():
            return False
        return True

    def wrapper(self, action): # nerual net should use this function directly
        if self.doAction(action):
            self.score += ACTION_REWARD
            return True, self.getRender()
        return False, self.score

    def getRender_new(self):
        id = self.current_piece.id

    def getRender(self):
        currentBoard = [[0 for _y in range(len(self.board[0]))] for _x in range(len(self.board))]
        shape = self.current_piece.getShape()
        dy, dx = self.current_piece_location[0], self.current_piece_location[1]
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x] == 1:
                    currentBoard[dy + y][dx + x] = 1
        ret = currentBoard + self.board
        return ret


    def getRender_(self):
        copyList = copy.deepcopy(self.board)
        shape = self.current_piece.getShape()
        dy, dx = self.current_piece_location[0], self.current_piece_location[1]
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x] == 1:
                    copyList[dy + y][dx + x] = 1
        return copyList

    # <<<< debug functions >>>>

    def play_debug(self):
        while True:
            action = int(input("give input"))
            self.doAction(action)
            self.display_debug()

    def display_debug(self):
        copyList = copy.deepcopy(self.board)
        shape = self.current_piece.getShape()
        dy, dx = self.current_piece_location[0], self.current_piece_location[1]
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x] == 1:
                    copyList[dy + y][dx + x] = 1
        print("_______________________")
        for line in range(len(copyList)):
            print(copyList[line])

