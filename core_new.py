from enum import Enum
import random
import copy

#<<Reward Functions>>
SURVIVAL_REWARD = 0
ACTION_REWARD = 0
TETRIS_REWARD = 1
functions = []

class Piece:

    def __init__(self):
        global functions
        self.orientation = 0
        self.shapes = []
        dict = {"I":self.I,"O":self.O, "T":self.T, "S":self.S, "Z":self.Z, "L":self.L, "R":self.R}
        dict_id = {"I": 0, "O": 1, "T": 2, "S": 3, "Z": 4, "L": 5, "R": 6}
        if len(functions) == 0:
            functions = ["I","O","T","S","Z","L","R"]
            random.shuffle(functions)
        temp = functions.pop()
        self.id = dict_id[temp]
        function = dict[temp]
        function()

    def I(self):
        self.shapes = [[[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],[[1, 1, 1, 1]]]

    def O(self):
        self.shapes = [[[1, 1], [1, 1]]]

    def T(self):
        self.shapes = [[[1, 0, 0], [1, 1, 0], [1, 0, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 1]],
                       [[0, 0, 1], [0, 1, 1], [0, 0, 1]]]

    def S(self):
        self.shapes = [[[1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 1], [1, 1, 0]]]

    def Z(self):
        self.shapes = [[[1, 0, 0], [1, 1, 0], [0, 1, 0]], [[1, 1, 0], [0, 1, 1]]]

    def L(self):
        self.shapes = [[[1,0],[1,0],[1,1]],[[1,0,0],[1,1,1]],[[1,1],[0,1],[0,1]],[[1,1,1],[0,0,1]]]

    def R(self):
        self.shapes = [[[0,1],[0,1],[1,1]],[[0,0,1],[1,1,1]],[[1,1],[1,0],[1,0]],[[1,1,1],[1,0,0]]]

    def rotate(self):
        if self.orientation == len(self.shapes) - 1:
            self.orientation = 0
        else:
            self.orientation += 1

    def setDefaultOrientation(self):
        self.orientation = 0

    def setOrientation(self, orientation):
        self.orientation = orientation

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
        self.leftActions = 0
        self.rotateActions = 0
        self.shouldContinue = True
        #self.playAIDebug()
        return

    def doAction(self, action, play=False): # when return false, game over
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
            if play and action == 0 and not self.checkTetris():
                 return False
            return False  # remember check this
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
        # shape = self.current_piece.getShape()
        # dy, dx = self.current_piece_location[0], self.current_piece_location[1]
        # for y in range(len(shape)):
        #     for x in range(len(shape[0])):
        #         if shape[y][x] == 1:
        #             self.board[dy + y][dx + x] = 1
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

    def doActionWithNumber(self, number):

        self.current_piece_location = [0, 0]
        self.current_piece.setDefaultOrientation()
        number_of_left = number // 4
        number_of_rotate = number % 4
        self.leftActions = int(number_of_left)
        self.rotateActions = int(number_of_rotate)

        while number_of_left:
            if not self.doAction(2):
                return False
            number_of_left -= 1
        while number_of_rotate:
            if not self.doAction(3):
                return False
            number_of_rotate -= 1
        self.shouldContinue = False
        self.current_piece_location = [0, 0]
        self.current_piece.setDefaultOrientation()
        return True

    def wrapper(self, action_list):

        if not self.shouldContinue:
            return True
        print(action_list)


        for action in action_list:
            if self.doActionWithNumber(action):
                break
        return True




    def getPossibleScenarios(self):
        candidates = []
        for x in range(len(self.board[0])):
            self.current_piece_location = [0,x]
            self.current_piece.setDefaultOrientation()
            if not self.checkObstruction(0,0):
                break
            for orientation in range(len(self.current_piece.shapes)):
                while self.doAction(0):
                    continue
                score = self.getFeatures(self.getRender_())
                candidates.append((score, self.getRender_(),(x, orientation)))
                self.current_piece_location = [0, x]
                if not self.doAction(3):
                    break
        candidates.sort(key=lambda x : x[0])
        best_candidate =  candidates[-1]
        return best_candidate

    def getRender_new(self):
        new_list = [0] * 7
        for i in range(7):
            if i == self.current_piece.id:
                new_list[i] = 1
        return [new_list] + self.board

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

    def playDebug(self):
        while True:
            action = int(input("give input"))
            self.doAction(action)
            self.display_debug()

    def playAIDebug(self):
        a = ""
        f = open("data3.txt", "a")
        while True:
            best_candidate = self.getPossibleScenarios()
            data = (self.getRender_new(), best_candidate[2][0] * 4 + best_candidate[2][1])
            self.board = best_candidate[1]
            f.write(str(data))
            f.write('\n')
            if not self.checkTetris():
                print(self.score)
                break
            if (self.score) % 100 == 0:
                print(self.score, "!")
            #self.displayDebug()

    def displayDebug(self):
        copyList = copy.deepcopy(self.board)
        # shape = self.current_piece.getShape()
        # dy, dx = self.current_piece_location[0], self.current_piece_location[1]
        # for y in range(len(shape)):
        #     for x in range(len(shape[0])):
        #         if shape[y][x] == 1:
        #             copyList[dy + y][dx + x] = 1
        print("_______________________")
        for line in range(len(copyList)):
            print(copyList[line])

    def nextStep(self):
        if self.shouldContinue == True:
            return
        if self.rotateActions > 0:
            self.rotateActions -=1
            self.doAction(3)
            return
        if self.leftActions > 0:
            self.leftActions -= 1
            self.doAction(2)
            return
        if not self.doAction(0):
            self.board = self.getRender_()
            self.shouldContinue = True
            if not self.checkTetris():
                return False


    # <<<< reward functions >>>>\

    def getFeatures(self,board):
        a = -0.510066
        b = 0.760666
        c = -0.35663
        d = -0.184483
        return self.getAggragateHeight(board) * a + self.getCompleteLines(board) * b + self.getHoles(board) * c + self.getBumpiness(board) * d

    def getColumnHeight(self, x, board):
        ret = 0
        for y in range(len(board)):
            if board[y][x] == 1:
                ret = max(ret, len(board)-y)
        return ret

    def getAggragateHeight(self, board):
        ret = 0
        for x in range(len(board[0])):
            ret += self.getColumnHeight(x, board)
        return ret

    def getBumpiness(self, board):
        ret = 0
        for x in range(len(board[0])-1):
            ret += abs(self.getColumnHeight(x, board) - self.getColumnHeight(x+1, board))
        return ret

    def getCompleteLines(self, board):
        ret = 0
        for y in range(len(board)):
            if sum(board[y]) == len(board[0]):
                ret += 1
        return ret

    def getHoles(self, board):
        ret = 0
        for x in range(len(board[0])):
            for y in range(1,len(board)):
                if board[y][x] == 0 and board[y-1][x] == 1:
                    ret += 1
        return ret
