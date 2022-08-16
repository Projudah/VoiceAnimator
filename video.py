import moviepy.editor as mpy
from helper import numberInDigits
from helper import removeAll


def getBaseClips():
    face_low = mpy.ImageClip('Data/Base Images/face low.png')
    face_mid = mpy.ImageClip('Data/Base Images/face mid.png')
    face_high = mpy.ImageClip('Data/Base Images/face high.png')
    hands_low = mpy.ImageClip('Data/Base Images/hands low.png')
    hands_mid = mpy.ImageClip('Data/Base Images/hands mid.png')
    hands_high = mpy.ImageClip('Data/Base Images/hands high.png')
    zoom_low = mpy.ImageClip('Data/Base Images/zoom low.png')
    zoom_mid = mpy.ImageClip('Data/Base Images/zoom mid.png')
    zoom_high = mpy.ImageClip('Data/Base Images/zoom high.png')
    return face_low, face_mid, face_high, hands_low, hands_mid, hands_high, zoom_low, zoom_mid, zoom_high


def getClipToDraw(value, allClips):
    face_low, face_mid, face_high, hands_low, hands_mid, hands_high, zoom_low, zoom_mid, zoom_high = allClips
    if value == "face_low":
        return face_low
    if value == "face_mid":
        return face_mid
    if value == "face_high":
        return face_high
    if value == "hands_low":
        return hands_low
    if value == "hands_mid":
        return hands_mid
    if value == "hands_high":
        return hands_high
    if value == "zoom_low":
        return zoom_low
    if value == "zoom_mid":
        return zoom_mid
    if value == "zoom_high":
        return zoom_high


def drawFrames(clip, start, count, fileName, numberOfDigits):
    for i in range(start, start+count):
        clip.save_frame(fileName+numberInDigits(i, numberOfDigits)+".jpg")


def createSequenceFiles(labels, classes, step, fps, folder):
    removeAll(folder)
    allClips = getBaseClips()

    fileName = folder + '/'

    framesToDraw = int(fps * step)

    totalFileCount = framesToDraw * len(labels)
    numDigits = len(str(totalFileCount))

    fileCount = 0
    print("FramestoDraw:", framesToDraw, "len:", len(labels))

    for label in labels:
        className = classes[int(label)]

        clip = getClipToDraw(className, allClips)
        drawFrames(clip, fileCount, framesToDraw, fileName, numDigits)

        fileCount = fileCount + framesToDraw


def saveSequenceAsVideo(folder, fps, fileName, audio=""):
    if audio != "":
        audioClip = mpy.AudioFileClip(audio)
    clip = mpy.ImageSequenceClip(folder, fps)
    if audioClip:
        clip = clip.set_audio(audioClip)
    clip.write_videofile(fileName, fps)
