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

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import random

# --------------------------------------------------------

class LabelWithLabel(QtGui.QWidget):
    def __init__(self, parent, labText='Label', defaultValue=''):
        super(LabelWithLabel, self).__init__(parent)
        self.label = QtGui.QLabel(defaultValue)
        self.labelsLabel = QtGui.QLabel(labText)
        self.initUI()

    def initUI(self):
        self.labelsLabel.setStyleSheet('font-size:10pt;')
        self.label.setStyleSheet('font-size:16pt; background-color:white; border:1px solid rgb(0, 0, 0); text-align:center')
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        vbox = QtGui.QVBoxLayout()
        vbox.setMargin(0)
        vbox.setSpacing(0)
        vbox.addWidget(self.labelsLabel)
        vbox.addWidget(self.label)
        self.setLayout(vbox)

# --------------------------------------------------------

class PlotWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.markedPoint = None
        self.markedPointData = [0, 0]
        self.canvas.mpl_connect('button_press_event', self.getXYDataOnClick)

        # self.button = QtGui.QPushButton('Plot FFT')
        # self.button.clicked.connect(self.plotRandom)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # layout.addWidget(self.button)
        self.setLayout(layout)

    def plotRandom(self):
        idxs = list(range(10))
        data = [ random.random() for i in range(10) ]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(idxs, data, '.-')
        self.canvas.draw()

    def plot(self, dataX, dataY, xlab='x', ylab='y'):
        self.figure.clear()
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.axis([ min(dataX)-0.5, max(dataX)+0.5, min(dataY)-0.5, max(dataY)+0.5 ])
        ax = self.figure.add_subplot(111)
        ax.plot(dataX, dataY, '.-')
        self.canvas.draw()

    def getXYDataOnClick(self, event):
        if self.markedPoint is not None:
            self.markedPoint.remove()
        self.markedPoint, = plt.plot(event.xdata, event.ydata, 'ro')
        self.markedPointData = [event.xdata, event.ydata]
        # latConst = np.abs(1.0 / event.xdata)
        # self.parent().latConstLabel.label.setText('{0:.2f} A'.format(latConst))
        # print('Rec. dist. = {0:.2f} A^-1'.format(event.xdata))
        # print('Lattice constant = {0:.2f} A'.format(latConst))

# --------------------------------------------------------

# def RunPlotWindow():
#     app = QtGui.QApplication(sys.argv)
#     main = PlotWidget()
#     main.show()
#     sys.exit(app.exec_())

# --------------------------------------------------------

class LatticeAnalyzerWidget(QtGui.QWidget):
    def __init__(self):
        super(LatticeAnalyzerWidget, self).__init__()
        self.display = QtGui.QLabel()
        imagePath = QtGui.QFileDialog.getOpenFileName()
        self.image = LoadImageSeriesFromFirstFile(imagePath)
        self.pointSets = []
        self.plotWidget = PlotWidget()
        self.latConstLabel = LabelWithLabel(self, 'Lattice constant', '0.0 A')
        self.createPixmap()
        self.initUI()

    def initUI(self):
        self.display.setFixedWidth(const.ccWidgetDim)
        self.display.setFixedHeight(const.ccWidgetDim)
        self.display.setAlignment(QtCore.Qt.AlignCenter)
        # self.plotWidget.setFixedWidth(const.ccWidgetDim)
        self.plotWidget.canvas.setFixedHeight(200)

        prevButton = QtGui.QPushButton('Prev', self)
        nextButton = QtGui.QPushButton('Next', self)
        clearButton = QtGui.QPushButton('Clear', self)
        startButton = QtGui.QPushButton('Start', self)      # Get freq. distribution (FFT)
        getLatConstButton = QtGui.QPushButton('Get lattice constant', self)

        prevButton.clicked.connect(partial(self.changePixmap, False))
        nextButton.clicked.connect(partial(self.changePixmap, True))
        clearButton.clicked.connect(self.clearImage)
        startButton.clicked.connect(self.calcHistogramWithRotation)
        getLatConstButton.clicked.connect(self.getLatConst)

        self.latConstLabel.setFixedHeight(100)

        hbox_nav = QtGui.QHBoxLayout()
        hbox_nav.addWidget(prevButton)
        hbox_nav.addWidget(nextButton)

        vbox_opt = QtGui.QVBoxLayout()
        vbox_opt.addWidget(clearButton)
        vbox_opt.addWidget(startButton)
        vbox_opt.addWidget(getLatConstButton)

        vbox_panel = QtGui.QVBoxLayout()
        vbox_panel.addLayout(hbox_nav)
        vbox_panel.addLayout(vbox_opt)
        vbox_panel.addWidget(self.latConstLabel)

        hbox_disp_and_panel = QtGui.QHBoxLayout()
        hbox_disp_and_panel.addWidget(self.display)
        hbox_disp_and_panel.addLayout(vbox_panel)

        vbox_main = QtGui.QVBoxLayout()
        vbox_main.addLayout(hbox_disp_and_panel)
        vbox_main.addWidget(self.plotWidget)
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

    def calcHistogramWithRotation(self):
        # Find center point of the line
        points = self.pointSets[self.image.numInSeries-1][:2]
        points = np.array([ CalcRealCoords(const.dimSize, pt) for pt in points ])
        rotCenter = np.average(points, 0).astype(np.int32)
        print('rotCenter = {0}'.format(rotCenter))

        # Find direction (angle) of the line
        # take sign of angle into account!!!
        dirAngles = FindDirectionAngle(points[0], points[1])
        print([ imsup.Degrees(da) for da in dirAngles ])

        # Shift image by -center
        shiftToRotCenter = list(-rotCenter)
        shiftToRotCenter.reverse()
        imgShifted = cc.ShiftImage(self.image, shiftToRotCenter)
        imgShifted = imsup.CreateImageWithBufferFromImage(imgShifted)

        # Rotate image by angle
        imgRot = tr.RotateImageSki2(imgShifted, imsup.Degrees(-dirAngles[0]))
        # imgRot = imsup.RotateImage(imgShifted, imsup.Degrees(-dirAngles[0]))
        imgRot.MoveToCPU()     # imgRot should already be stored in CPU memory

        # Crop fragment whose height = distance between two points
        ptDiffs = points[0]-points[1]
        fragHeight = int(np.sqrt(ptDiffs[0] ** 2 + ptDiffs[1] ** 2))
        fragWidth = fragHeight
        print('Frag height = {0}'.format(fragHeight))
        fragCoords = imsup.DetermineCropCoordsForNewWidth(imgRot.width, fragWidth)
        print('Frag coords = {0}'.format(fragCoords))
        imgCropped = imsup.CreateImageWithBufferFromImage(imsup.CropImageROICoords(imgRot, fragCoords))
        imgCropped.MoveToCPU()

        imgList = imsup.ImageList([self.image, imgShifted, imgRot, imgCropped])
        imgList.UpdateLinks()

        # Calculate projection of intensity and FFT
        distances = np.arange(0, fragWidth, 1, np.float32)
        distances *= const.pxWidth
        intMatrix = np.copy(imgCropped.amPh.am)
        intProjection = np.sum(intMatrix, 0)      # 0 - horizontal projection, 1 - vertical projection
        intProjFFT = list(np.fft.fft(intProjection))
        arrHalf = len(intProjFFT) // 2
        intProjFFT = np.array(intProjFFT[arrHalf:] + intProjFFT[:arrHalf])
        intProjFFTReal, intProjFFTImag = np.abs(np.real(intProjFFT)), np.imag(intProjFFT)

        recPxWidth = 1.0 / (intProjection.shape[0] * const.pxWidth)
        recOrigin = -1.0 / (2.0 * const.pxWidth)
        recDistances = np.array([ recOrigin + x * recPxWidth for x in range(intProjection.shape[0]) ])

        intProjFFTRealToPlot = imsup.ScaleImage(intProjFFTReal, 0, 10)
        recDistsToPlot = recDistances * 1e-10     # A^-1
        self.plotWidget.plot(recDistsToPlot, intProjFFTRealToPlot, 'rec. dist. [1/A]', 'FFT [a.u.]')

        # Rysuj widmo FFT w GUI
        # Użytkownik zaznacza odpowiednie maksimum
        # Do piku dopasowywany jest Gauss
        # Wyznaczana jest odpowiednia stała sieci

        # file1 = open('int_proj.txt', 'w')
        # for dist, intProj in zip(distances, intProjection):
        #     file1.write('{0:.2f}\t{1:.2f}\n'.format(dist * 1e10, intProj))
        # file1.close()
        #
        # file2 = open('int_proj_fft.txt', 'w')
        # for dist, intProj in zip(recDistances, intProjFFTReal):
        #     file2.write('{0:.2f}\t{1:.2f}\n'.format(dist * 1e-10, intProj))
        # file2.close()

    def getLatConst(self):
        recDist = self.plotWidget.markedPointData[0]
        latConst = np.abs(1.0 / recDist)
        self.latConstLabel.label.setText('{0:.2f} A'.format(latConst))

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

def FindDirectionAngle(p1, p2):
    ang1 = np.arctan2(np.abs(p1[0]-p2[0]), np.abs(p1[1]-p2[1]))
    ang2 = np.pi / 2 - ang1
    return ang1, ang2

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