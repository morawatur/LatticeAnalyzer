import re
import sys
from os import path
from functools import partial
import numpy as np
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

class LineEditWithLabel(QtGui.QWidget):
    def __init__(self, parent, labText='Label', defaultValue=''):
        super(LineEditWithLabel, self).__init__(parent)
        self.label = QtGui.QLabel(labText)
        self.input = QtGui.QLineEdit(defaultValue)
        self.initUI()

    def initUI(self):
        self.input.setMaxLength(10)
        vbox = QtGui.QVBoxLayout()
        vbox.setMargin(0)
        vbox.setSpacing(0)
        vbox.addWidget(self.label)
        vbox.addWidget(self.input)
        self.setLayout(vbox)

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

# zrobic po prostu LabelWithImage?
class LabelExt(QtGui.QLabel):
    def __init__(self, parent, image=None):
        super(LabelExt, self).__init__(parent)
        self.image = image
        self.setImage()
        self.pointSets = [[]]
        # while image.next is not None:
        #    self.pointSets.append([])

    def paintEvent(self, event):
        super(LabelExt, self).paintEvent(event)
        linePen = QtGui.QPen(QtCore.Qt.white)
        linePen.setCapStyle(QtCore.Qt.RoundCap)
        linePen.setWidth(2)
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        qp.setPen(linePen)
        imgIdx = self.image.numInSeries - 1
        intWidth = int(int(self.parent().intWidthInput.input.text()) * (self.width() / self.image.width))
        for pt in self.pointSets[imgIdx]:
            rect = QtCore.QRect(pt[0]-4, pt[1]-4, 9, 9)
            qp.drawArc(rect, 0, 16*360)
        for pt1, pt2 in zip(self.pointSets[imgIdx][:-1], self.pointSets[imgIdx][1:]):
            line = QtCore.QLine(pt1[0], pt1[1], pt2[0], pt2[1])
            qp.drawLine(line)
            rect = CalcRectVertices(pt1, pt2, intWidth)
            for vert1, vert2 in zip(rect, rect[1:] + [rect[0]]):
                line = QtCore.QLine(vert1[0], vert1[1], vert2[0], vert2[1])
                qp.drawLine(line)
        qp.end()

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        currPos = [pos.x(), pos.y()]
        self.pointSets[self.image.numInSeries - 1].append(currPos)
        self.repaint()

    def setImage(self):
        paddedImage = imsup.PadImageBufferToNx512(self.image, np.max(self.image.buffer))
        qImg = QtGui.QImage(imsup.ScaleImage(paddedImage.buffer, 0.0, 255.0).astype(np.uint8),
                            paddedImage.width, paddedImage.height, QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(qImg)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)
        self.setPixmap(pixmap)
        self.repaint()

    def changeImage(self, toNext=True):
        newImage = self.image.next if toNext else self.image.prev
        if newImage is not None:
            newImage.ReIm2AmPh()
            self.image = newImage
            if len(self.pointSets) < self.image.numInSeries:
                self.pointSets.append([])
            self.setImage()

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
        self.markedPoint = None
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

# --------------------------------------------------------

class LatticeAnalyzerWidget(QtGui.QWidget):
    def __init__(self):
        super(LatticeAnalyzerWidget, self).__init__()
        imagePath = QtGui.QFileDialog.getOpenFileName()
        image = LoadImageSeriesFromFirstFile(imagePath)
        self.display = LabelExt(self, image)
        self.plotWidget = PlotWidget()
        self.intWidthInput = LineEditWithLabel(self, 'Integration width [px]', '50')
        self.latConstLabel = LabelWithLabel(self, 'Lattice constant', '0.0 A')
        self.initUI()

    def initUI(self):
        self.display.setFixedWidth(const.ccWidgetDim)
        self.display.setFixedHeight(const.ccWidgetDim)
        self.display.setAlignment(QtCore.Qt.AlignCenter)
        # self.plotWidget.setFixedWidth(const.ccWidgetDim)
        self.plotWidget.canvas.setFixedHeight(200)

        prevButton = QtGui.QPushButton('Prev', self)
        nextButton = QtGui.QPushButton('Next', self)
        startButton = QtGui.QPushButton('Start', self)      # Get freq. distribution (FFT)
        zoomButton = QtGui.QPushButton('Zoom', self)
        clearButton = QtGui.QPushButton('Clear', self)
        removeButton = QtGui.QPushButton('Delete', self)
        exportButton = QtGui.QPushButton('Export', self)
        getLatConstButton = QtGui.QPushButton('Get lattice constant', self)

        prevButton.clicked.connect(self.goToPrevImage)
        nextButton.clicked.connect(self.goToNextImage)
        startButton.clicked.connect(self.calcHistogramWithRotation)
        zoomButton.clicked.connect(self.zoomImage)
        clearButton.clicked.connect(self.clearImage)
        # removeButton.clicked.connect(self.removeImage)
        exportButton.clicked.connect(self.exportImage)
        getLatConstButton.clicked.connect(self.getLatConst)

        self.intWidthInput.setFixedHeight(1.5 * startButton.height())
        self.latConstLabel.setFixedHeight(100)

        hbox_nav = QtGui.QHBoxLayout()
        hbox_nav.addWidget(prevButton)
        hbox_nav.addWidget(nextButton)

        vbox_opt = QtGui.QVBoxLayout()
        vbox_opt.addWidget(self.intWidthInput)
        vbox_opt.addWidget(startButton)
        vbox_opt.addWidget(zoomButton)
        vbox_opt.addWidget(clearButton)
        vbox_opt.addWidget(removeButton)
        vbox_opt.addWidget(exportButton)
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

    def goToPrevImage(self):
        self.display.changeImage(toNext=False)

    def goToNextImage(self):
        self.display.changeImage(toNext=True)

    # prowizorka
    def zoomImage(self):
        imgIdx = self.display.image.numInSeries - 1
        pt1, pt2 = self.display.pointSets[imgIdx][:2]
        lpt = pt1[:] if pt1[0] < pt2[0] else pt2[:]     # left point
        rpt = pt1[:] if pt1[0] > pt2[0] else pt2[:]     # right point
        currImage = self.display.image
        lpt = CalcRealTLCoordsForPaddedImage(self.display.image.width, lpt)
        rpt = CalcRealTLCoordsForPaddedImage(self.display.image.width, rpt)
        width = np.abs(rpt[0] - lpt[0])
        height = np.abs(rpt[1] - lpt[1])
        biggerDim = width if width > height else height
        zoomCoords = [lpt[0], lpt[1], lpt[0] + biggerDim, lpt[1] + biggerDim]
        # a moze uzyc RescaleSkiImage?
        zoom = imsup.CropImgAmpFragment(self.display.image, zoomCoords)
        zoom.prev = currImage
        if currImage.next is not None:
            zoom.next = currImage.next
            currImage.next.prev = zoom
        currImage.next = zoom
        self.goToNextImage()
        self.clearImage()

    def calcHistogramWithRotation(self):
        # Find center point of the line
        imgIdx = self.display.image.numInSeries - 1
        points = self.display.pointSets[imgIdx][:2]
        # points = np.array([ CalcRealCoords(self.display.image.width, pt) for pt in points ])
        points = np.array([ CalcRealMidCoordsForPaddedImage(self.display.image.width, pt) for pt in points ])
        rotCenter = np.average(points, 0).astype(np.int32)
        print('rotCenter = {0}'.format(rotCenter))

        # Find direction (angle) of the line
        dirInfo = FindDirectionAngles(points[0], points[1])
        dirAngle = imsup.Degrees(dirInfo[0])
        projDir = dirInfo[2]
        print('dirAngle = {0:.2f} deg'.format(dirAngle))

        # Shift image by -center
        shiftToRotCenter = list(-rotCenter)
        shiftToRotCenter.reverse()
        imgShifted = cc.ShiftImage(self.display.image, shiftToRotCenter)
        imgShifted = imsup.CreateImageWithBufferFromImage(imgShifted)

        # Rotate image by angle
        imgRot = tr.RotateImageSki2(imgShifted, dirAngle)
        # imgRot = imsup.RotateImage(imgShifted, dirAngle)
        imgRot.MoveToCPU()     # imgRot should already be stored in CPU memory

        # Crop fragment whose height = distance between two points
        ptDiffs = points[0]-points[1]
        fragDim1 = int(np.sqrt(ptDiffs[0] ** 2 + ptDiffs[1] ** 2))
        fragDim2 = int(self.intWidthInput.input.text())
        if projDir == 0:
            fragWidth, fragHeight = fragDim1, fragDim2
        else:
            fragWidth, fragHeight = fragDim2, fragDim1

        fragCoords = imsup.DetermineCropCoordsForNewDims(imgRot.width, imgRot.height, fragWidth, fragHeight)
        print('Frag dims = {0}, {1}'.format(fragWidth, fragHeight))
        print('Frag coords = {0}'.format(fragCoords))
        imgCropped = imsup.CropImgAmpFragment(imgRot, fragCoords)

        # squareFragCoords = imsup.DetermineCropCoordsForNewWidth(imgRot.width, fragDim1)
        # imgCroppedToDisp = imsup.CropImgAmpFragment(imgRot, squareFragCoords)
        # imgList = imsup.ImageList([self.display.image, imgShifted, imgRot, imgCroppedToDisp])
        # imgList.UpdateLinks()

        # Calculate projection of intensity and FFT
        distances = np.arange(0, fragWidth, 1, np.float32)
        distances *= const.pxWidth
        intMatrix = np.copy(imgCropped.amPh.am)
        print(projDir)
        intProjection = np.sum(intMatrix, projDir)      # 0 - horizontal projection, 1 - vertical projection
        intProjFFT = list(np.fft.fft(intProjection))
        arrHalf = len(intProjFFT) // 2
        intProjFFT = np.array(intProjFFT[arrHalf:] + intProjFFT[:arrHalf])
        intProjFFTReal, intProjFFTImag = np.abs(np.real(intProjFFT)), np.imag(intProjFFT)

        numOfNegDists = arrHalf
        if intProjection.shape[0] % 2:
            numOfNegDists += 1
        recPxWidth = 1.0 / (intProjection.shape[0] * const.pxWidth)
        recOrigin = -numOfNegDists * recPxWidth
        recDistances = np.array([recOrigin + x * recPxWidth for x in range(intProjection.shape[0])])

        intProjFFTRealToPlot = imsup.ScaleImage(intProjFFTReal, 0, 10)
        recDistsToPlot = recDistances * 1e-10     # A^-1
        self.plotWidget.plot(recDistsToPlot, intProjFFTRealToPlot, 'rec. dist. [1/A]', 'FFT [a.u.]')

    def getLatConst(self):
        recDist = self.plotWidget.markedPointData[0]
        latConst = np.abs(1.0 / recDist)
        self.latConstLabel.label.setText('{0:.2f} A'.format(latConst))

    def clearImage(self):
        imgIdx = self.display.image.numInSeries - 1
        self.display.pointSets[imgIdx][:] = []
        self.display.repaint()

    # def removeImage(self):
    #     self.clearImage()
    #     currImg = self.display.image
    #     prevImg = currImg.prev
    #     nextImg = currImg.next
    #     prevImg.next = nextImg
    #     nextImg.prev = prevImg
    #     nextImg.numInSeries -= 1
    #     del currImg

    def exportImage(self):
        fName = 'img{0}.png'.format(self.display.image.numInSeries)
        imsup.SaveAmpImage(self.display.image, fName)
        print('Saved image as "{0}"'.format(fName))

# --------------------------------------------------------

def LoadImageSeriesFromFirstFile(imgPath):
    imgList = imsup.ImageList()
    imgNumMatch = re.search('([0-9]+).dm3', imgPath)
    imgNumText = imgNumMatch.group(1)
    imgNum = int(imgNumText)

    while path.isfile(imgPath):
        print('Reading file "' + imgPath + '"')
        imgData = dm3.ReadDm3File(imgPath)
        imgDimSize = np.sqrt(len(imgData))
        imgMatrix = imsup.PrepareImageMatrix(imgData, imgDimSize)
        img = imsup.ImageWithBuffer(imgDimSize, imgDimSize, imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'])
        img.LoadAmpData(np.sqrt(imgMatrix).astype(np.float32))
        # ---
        imsup.RemovePixelArtifacts(img, const.minPxThreshold, const.maxPxThreshold)
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

def CalcRealMidCoordsForPaddedImage(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    padImgWidthReal = np.ceil(imgWidth / 512.0) * 512.0
    pad = (padImgWidthReal - imgWidth) / 2.0
    factor = padImgWidthReal / dispWidth
    realCoords = [ int(dc * factor - pad - imgWidth // 2) for dc in dispCoords ]
    return realCoords

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

def CalcDispCoords(imgWidth, realCoords):
    dispWidth = const.ccWidgetDim
    factor = dispWidth / imgWidth
    dispCoords = [ (rc * factor) + dispWidth // 2 for rc in realCoords ]
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

def FindDirectionAngles(p1, p2):
    lpt = p1[:] if p1[0] < p2[0] else p2[:]     # left point
    rpt = p1[:] if p1[0] > p2[0] else p2[:]     # right point
    dx = np.abs(rpt[0] - lpt[0])
    dy = np.abs(rpt[1] - lpt[1])
    sign = 1 if rpt[1] < lpt[1] else -1
    projDir = 1         # projection on y axis
    if dx > dy:
        sign *= -1
        projDir = 0     # projection on x axis
    diff1 = dx if dx < dy else dy
    diff2 = dx if dx > dy else dy
    ang1 = np.arctan2(diff1, diff2)
    ang2 = np.pi / 2 - ang1
    ang1 *= sign
    ang2 *= (-sign)
    return ang1, ang2, projDir

# --------------------------------------------------------

def CalcRectVertices(p1, p2, rectWidth):
    dirInfo = FindDirectionAngles(p1, p2)
    dirAngle = np.abs(dirInfo[0])
    projDir = dirInfo[2]
    d1 = np.abs((rectWidth / 2.0) * np.sin(dirAngle))
    d2 = np.abs((rectWidth / 2.0) * np.cos(dirAngle))
    if projDir == 0:
        dx, dy = d1, d2
    else:
        dx, dy = d2, d1
    lpt = p1[:] if p1[0] < p2[0] else p2[:]
    rpt = p1[:] if p1[0] > p2[0] else p2[:]
    vect = np.array([dx, dy]).astype(np.int32)
    p1 = np.array(p1).astype(np.int32)
    p2 = np.array(p2).astype(np.int32)
    if rpt[1] > lpt[1]: vect[1] *= -1
    rect = [ p1-vect, p1+vect, p2+vect, p2-vect ]
    rect = [ list(pt) for pt in rect ]
    return rect

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