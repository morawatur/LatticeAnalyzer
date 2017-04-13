import re
import sys
from os import path
from functools import partial
import numpy as np
from skimage import transform as tf
from PyQt4 import QtGui, QtCore
import Dm3Reader3 as dm3
import Constants as const
import ImageSupport as imsup
import CrossCorr as cc
import Transform as tr

# --------------------------------------------------------

class LatticeAnalyzerWidget(QtGui.QWidget):
    def __init__(self):
        super(LatticeAnalyzerWidget, self).__init__()
        self.display = QtGui.QLabel()
        imagePath = QtGui.QFileDialog.getOpenFileName()
        self.image = LoadImageSeriesFromFirstFile(imagePath)
        self.pointSets = []
        self.createPixmap()
        self.initUI()

    def initUI(self):
        prevButton = QtGui.QPushButton('Prev', self)
        nextButton = QtGui.QPushButton('Next', self)
        clearButton = QtGui.QPushButton('Clear', self)
        startButton = QtGui.QPushButton('Calculate', self)

        prevButton.clicked.connect(partial(self.changePixmap, False))
        nextButton.clicked.connect(partial(self.changePixmap, True))
        clearButton.clicked.connect(self.clearImage)
        startButton.clicked.connect(self.calcHistogram)

        hbox_nav = QtGui.QHBoxLayout()
        hbox_nav.addWidget(prevButton)
        hbox_nav.addWidget(nextButton)

        hbox_opt = QtGui.QHBoxLayout()
        hbox_opt.addWidget(clearButton)
        hbox_opt.addWidget(startButton)

        vbox_main = QtGui.QVBoxLayout()
        vbox_main.addWidget(self.display)
        vbox_main.addLayout(hbox_nav)
        vbox_main.addLayout(hbox_opt)
        self.setLayout(vbox_main)

        # self.statusBar().showMessage('Ready')
        self.move(250, 5)
        self.setWindowTitle('Lattice analyzer')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())      # disable window resizing

    def createPixmap(self):
        paddedImage = imsup.PadImageBufferToNx512(self.image, np.max(self.image.buffer))
        qImg = QtGui.QImage(imsup.ScaleImage(paddedImage.buffer, 0.0, 255.0).astype(np.uint8),
                            paddedImage.width, paddedImage.height, QtGui.QImage.Format_Indexed8)
        # qImg = QtGui.QImage(imsup.ScaleImage(self.image.buffer, 0.0, 255.0).astype(np.uint8),
        #                     self.image.width, self.image.height, QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(qImg)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)    # !!!
        self.display.setPixmap(pixmap)

    def changePixmap(self, toNext=True):
        newImage = self.image.next if toNext else self.image.prev
        labToDel = self.display.children()
        for child in labToDel:
            child.deleteLater()
            # child.hide()
        if newImage is not None:
            newImage.ReIm2AmPh()
            self.image = newImage
            self.createPixmap()
            if len(self.pointSets) < self.image.numInSeries:
                return
            for pt, idx in zip(self.pointSets[self.image.numInSeries-1], range(1, len(self.pointSets[self.image.numInSeries-1])+1)):
                print(self.image.numInSeries, pt)
                lab = QtGui.QLabel('{0}'.format(idx), self.display)
                lab.setStyleSheet('font-size:18pt; background-color:white; border:1px solid rgb(0, 0, 0);')
                lab.move(pt[0], pt[1])
                lab.show()

    # def mousePressEvent(self, QMouseEvent):
    #     print(QMouseEvent.pos())

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        currPos = [pos.x(), pos.y()]
        startPos = [ self.display.pos().x(), self.display.pos().y() ]
        endPos = [ startPos[0] + self.display.width(), startPos[1] + self.display.height() ]

        if startPos[0] < currPos[0] < endPos[0] and startPos[1] < currPos[1] < endPos[1]:
            currPos = [ a - b for a, b in zip(currPos, startPos) ]
            if len(self.pointSets) < self.image.numInSeries:
                self.pointSets.append([])
            self.pointSets[self.image.numInSeries-1].append(currPos)
            lab = QtGui.QLabel('{0}'.format(len(self.pointSets[self.image.numInSeries-1])), self.display)
            lab.setStyleSheet('font-size:18pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(currPos[0], currPos[1])
            lab.show()

    def calcHistogram(self):
        rectPoints = [ CalcRealCoords(const.dimSize, self.pointSets[0][pIdx]) for pIdx in range(3) ]
        A = rectPoints[0]
        B = rectPoints[1]
        C = rectPoints[2]
        D = [ A[0] + C[0] - B[0], A[1] + C[1] - B[1] ]
        rectPoints.append(D)
        for pt in rectPoints:
            print(pt)

        # A = [-10, -10]
        # B = [-5, 10]
        # C = [15, 8]
        # D = [10, -12]

        kAB = tr.Line(0, 0)
        lBC = tr.Line(0, 0)
        kCD = tr.Line(0, 0)
        lAD = tr.Line(0, 0)

        kAB.getFromPoints(A, B)
        lBC.getFromPoints(B, C)
        kCD.getFromPoints(C, D)
        lAD.getFromPoints(A, D)

        img = self.image
        a = 0

        for py in range(img.height):
            for px in range(img.width):
                pt = CalcNewCoords([px, py], [img.height // 2, img.width // 2])
                lTmp = tr.FindParallelLine(lBC, pt)
                pt1 = tr.FindCommonPoint(kAB, lTmp)
                pt2 = tr.FindCommonPoint(kCD, lTmp)
                if (pt1[0] < pt[0] < pt2[0]) and (pt2[1] < pt[1] < pt1[1]):
                    img.amPh.am[py, px] = 0.0

        imsup.DisplayAmpImage(img)
        print(a)
        print('All done!')

    def calcHistogramWithRotation(self):
        pass

    def clearImage(self):
        labToDel = self.display.children()
        for child in labToDel:
            child.deleteLater()
        self.pointSets[self.image.numInSeries - 1][:] = []

    # def exportImage(self):
    #     fName = 'img{0}.png'.format(self.image.numInSeries)
    #     imsup.SaveAmpImage(self.image, fName)
    #     print('Saved image as "{0}"'.format(fName))

# --------------------------------------------------------

def LoadImageSeriesFromFirstFile(imgPath):
    imgList = imsup.ImageList()
    imgNumMatch = re.search('([0-9]+).dm3', imgPath)
    imgNumText = imgNumMatch.group(1)
    imgNum = int(imgNumText)

    while path.isfile(imgPath):
        print('Reading file "' + imgPath + '"')
        imgData = dm3.ReadDm3File(imgPath)
        imgMatrix = imsup.PrepareImageMatrix(imgData, const.dimSize)
        img = imsup.ImageWithBuffer(const.dimSize, const.dimSize, imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'])
        img.LoadAmpData(np.sqrt(imgMatrix).astype(np.float32))
        # ---
        # imsup.RemovePixelArtifacts(img, const.minPxThreshold, const.maxPxThreshold)
        img.UpdateBuffer()
        # ---
        img.numInSeries = imgNum
        imgList.append(img)

        imgNum += 1
        imgNumTextNew = imgNumText.replace(str(imgNum-1), str(imgNum))
        if imgNum == 10:
            imgNumTextNew = imgNumTextNew[1:]
        imgPath = RReplace(imgPath, imgNumText, imgNumTextNew, 1)
        imgNumText = imgNumTextNew

    imgList.UpdateLinks()
    return imgList[0]

# --------------------------------------------------------

def CalcTopLeftCoords(imgWidth, midCoords):
    topLeftCoords = [ mc + imgWidth // 2 for mc in midCoords ]
    return topLeftCoords

# --------------------------------------------------------

def CalcTopLeftCoordsForSetOfPoints(imgWidth, points):
    topLeftPoints = [ CalcTopLeftCoords(imgWidth, pt) for pt in points ]
    return topLeftPoints

# --------------------------------------------------------

def CalcRealCoords(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    factor = imgWidth / dispWidth
    realCoords = [ int((dc - dispWidth // 2) * factor) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcRealCoordsForSetOfPoints(imgWidth, points):
    realPoints = [ CalcRealCoords(imgWidth, pt) for pt in points ]
    return realPoints

# --------------------------------------------------------

def CalcRealTLCoordsForPaddedImage(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    padImgWidthReal = np.ceil(imgWidth / 512.0) * 512.0
    pad = (padImgWidthReal - imgWidth) / 2.0
    factor = padImgWidthReal / dispWidth
    # dispPad = pad / factor
    # realCoords = [ (dc - dispPad) * factor for dc in dispCoords ]
    realCoords = [ int(dc * factor - pad) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcDispCoords(dispWidth, realCoords):
    imgWidth = const.dimSize
    factor = dispWidth / imgWidth
    dispCoords = [ (rc * factor) + const.ccWidgetDim // 2 for rc in realCoords ]
    return dispCoords

# --------------------------------------------------------

def CalcDistance(p1, p2):
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist

# --------------------------------------------------------

def CalcInnerAngle(a, b, c):
    alpha = np.arccos(np.abs((a*a + b*b - c*c) / (2*a*b)))
    return imsup.Degrees(alpha)

# --------------------------------------------------------

def CalcOuterAngle(p1, p2):
    dist = CalcDistance(p1, p2)
    betha = np.arcsin(np.abs(p1[0] - p2[0]) / dist)
    return imsup.Degrees(betha)

# --------------------------------------------------------

def CalcNewCoords(p1, newCenter):
    p2 = [ px - cx for px, cx in zip(p1, newCenter) ]
    return p2

# --------------------------------------------------------

# tu jeszcze cos nie tak (03-04-2017)
def CalcRotAngle(p1, p2):
    z1 = np.complex(p1[0], p1[1])
    z2 = np.complex(p2[0], p2[1])
    phi1 = np.angle(z1)
    phi2 = np.angle(z2)
    rotAngle = np.abs(imsup.Degrees(phi2 - phi1))
    # if rotAngle < 0:
    #     rotAngle = 360 - np.abs(rotAngle)
    return rotAngle

# --------------------------------------------------------

def SwitchXY(xy):
    return [xy[1], xy[0]]

# --------------------------------------------------------

def RReplace(text, old, new, occurence):
    rest = text.rsplit(old, occurence)
    return new.join(rest)

# --------------------------------------------------------

def RunLatticeAnalyzer():
    app = QtGui.QApplication(sys.argv)
    laWindow = LatticeAnalyzerWidget()
    sys.exit(app.exec_())