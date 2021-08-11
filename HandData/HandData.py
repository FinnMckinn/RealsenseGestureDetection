import numpy as np

class FingerData:

    def __init__(self, colour):
        self.tipPoint = None
        self.knucklePoint = None
        self.colour = colour

    def setTipPoint(self, cord):
        self.tipPoint = cord

    def setknucklePoint(self, cord):
        self.knucklePoint = cord


class PalmData:

    def __init__(self, colour):
        self.centrePoint = None
        self.colour = colour

        self.lastCords = []
        self.averagePoint = (0,0)        

    def setCentrePoint(self, cord):
        self.centrePoint = cord
        if np.count_nonzero(cord) > 0:
            if len(self.lastCords) == 5:
                self.lastCords.pop(0)
            self.lastCords.append(cord)
    
    def setcolour(self, colour):
        self.colour = colour

    def calcAverageCord(self):
        if len(self.lastCords) > 0:
            averageX = 0
            averageY = 0

            for cords in self.lastCords:
                averageX += cords[0]
                averageY += cords[1]
            
            averageX = int(averageX/len(self.lastCords))
            averageY = int(averageY/len(self.lastCords))

            self.averagePoint = (averageX, averageY)

