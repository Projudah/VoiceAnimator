import numpy
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioTrainTest as aT
import os.path
from helper import removeAll
import video
import audio
import glob

DataWav = 'Data\Data1.wav'
DataSeg = 'Data\Data1.segments'

DataReg = 'Data/RegressionData'

TestWav = 'Data/test2.wav'
TestSeg = 'Data/test1.segments'

midStep = 1
fps = 10

regStep = 1


def saveToFile(labels, classes, step, fileName):
    string = ''
    start = 0
    for label in labels:
        className = classes[label]
        string += "%s\t%s\t%s\n" % (start, start + step, className)
        start = start + step
    with open(fileName, 'w') as file:
        file.write(string)


def regressToLabels(results, classes):
    labels = []

    for data in results:
        index = max(range(len(data)), key=data.__getitem__)
        labels.append(index)
    return labels, classes


def tryhmm():
    aS.train_hmm_from_file(DataWav, DataSeg,
                           'hmmTemp1', 0.5, 0.5)  # train using a single file

    labels, class_names, accuracy, cm = aS.hmm_segmentation(
        TestWav, 'hmmTemp1', True, TestSeg)

    # test 1
    print(labels, class_names, accuracy, cm)


def tryclassify():
    modelExists = os.path.exists('judahSpeakSvm')

    if not modelExists:
        aT.extract_features_and_train(["Data\ClassifyTrain\high", "Data\ClassifyTrain\low",
                                       "Data\ClassifyTrain\mid"], midStep, midStep, aT.shortTermWindow, aT.shortTermStep, "svm", "judahSpeakSvm")

    # labels, classesAll, acc, CM = aS.mid_term_file_classification(
    #     TestWav, "judahSpeakSvm", "svm", True, TestSeg)
    labels, classesAll, acc, CM = aS.mid_term_file_classification(
        TestWav, "judahSpeakSvm", "svm")

    saveToFile(labels.astype(numpy.int64), classesAll, midStep, "output.tsv")

    print(labels, classesAll, acc, CM)

    video.createSequenceFiles(labels, classesAll, midStep, fps, "Data/Staging")

    video.saveSequenceAsVideo(
        "Data/Staging", fps, 'testVideoClassify.mp4', TestWav)
    removeAll('Data/Staging')


def tryregress():
    modelExists = len(glob.glob('judahSpeakReg*')) > 0

    if not modelExists:
        aT.feature_extraction_train_regression(
            DataReg, regStep, regStep, aT.shortTermWindow, aT.shortTermStep, "svm", "judahSpeakReg", False)

    values, classes = regressFile(TestWav, 'judahSpeakReg')

    print(classes, values)

    video.createSequenceFiles(values, classes, regStep, fps, "Data/Staging")

    video.saveSequenceAsVideo("Data/Staging", fps, 'testVideoReg.mp4', TestWav)
    removeAll('Data/Staging')


def regressFile(audioFile, model_name):
    audio.segmentWavToFolder(audioFile, regStep, 'Data/RegressionSegments')
    results, labels = audio.regressionFolderWrapper(
        "Data/RegressionSegments", "svm", model_name)

    print(results, labels)
    removeAll('Data/RegressionSegments')

    return regressToLabels(results, labels)


tryregress()
tryclassify()
