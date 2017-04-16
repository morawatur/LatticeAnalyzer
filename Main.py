from numba import cuda
import Constants as const
import GUI2 as gui
import ImageSupport as imsup
import CrossCorr as cc
import time

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

gui.RunLatticeAnalyzer()