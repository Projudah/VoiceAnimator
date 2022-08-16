from pyAudioAnalysis import audioBasicIO
import scipy.io.wavfile as wavfile

import os
import numpy
import glob
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioTrainTest as aT
from helper import removeAll
from helper import numberInDigits
import math


def isMoreThanShortTerm(sampling, signalArray):
    window = round(sampling * aT.shortTermWindow)
    return len(signalArray) > window


def segmentWavToFolder(wavFile, steps, outputFolder):
    removeAll(outputFolder)
    [Fs, x] = audioBasicIO.read_audio_file(wavFile)

    divideBy = int(round(steps * Fs))
    numDigits = len(str(math.ceil(len(x)/divideBy)))
    seconds = 0
    formatedSecs = 0
    while formatedSecs < len(x):
        time1 = int(round(seconds*Fs))
        time2 = int(round((seconds + steps)*Fs))
        # print("seconds", seconds, "time1:", time1,
        #       "time2:", time2, "len:", len(x))
        if time2 >= len(x):
            time2 = len(x)-1
        xtemp = x[time1:time2]
        if isMoreThanShortTerm(Fs, xtemp):
            wavfile.write(outputFolder+"/out_" +
                          numberInDigits(seconds, numDigits)+".wav", Fs, xtemp)
        formatedSecs = int(round((seconds + steps)*Fs))
        seconds += steps


def regressionFolderWrapper(inputFolder, model_type, model_name):
    files = "*.wav"
    if os.path.isdir(inputFolder):
        strFilePattern = os.path.join(inputFolder, files)
    else:
        strFilePattern = inputFolder + files

    wavFilesList = []
    wavFilesList.extend(glob.glob(strFilePattern))
    wavFilesList = sorted(wavFilesList)
    if len(wavFilesList) == 0:
        print("No WAV files found!")
        return
    Results = []
    for wavFile in wavFilesList:
        R, regressionNames = aT.file_regression(
            wavFile, model_name, model_type)
        Results.append(R)
        print("Regressing: ", wavFile, R)
    print("Names: ", regressionNames)
    NewResults = numpy.array(Results)

    for i, r in enumerate(regressionNames):
        [Histogram, bins] = numpy.histogram(NewResults[:, i])
        centers = (bins[0:-1] + bins[1::]) / 2.0
        plt.subplot(len(regressionNames), 1, i + 1)
        plt.plot(centers, Histogram)
        plt.title(r)
    plt.show()
    return Results, regressionNames
