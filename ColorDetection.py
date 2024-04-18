import cv2
import numpy as np

cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars',640,240)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass
cv2.createTrackbar('hue min','TrackBars',0,179,empty)
cv2.createTrackbar('hue max','TrackBars',179,179,empty)#色相
cv2.createTrackbar('sat min','TrackBars',0,255,empty)
cv2.createTrackbar('sat max','TrackBars',255,255,empty)#饱和度
cv2.createTrackbar('val min','TrackBars',0,255,empty)
cv2.createTrackbar('val max','TrackBars',255,255,empty)#明度

vc = cv2.VideoCapture('rawdata\BV17d4y1171Y\BV17d4y1171Y-1.mp4')
if  not vc.isOpened():
    print("Failed to open the video file.")
    exit()
pause = False

while True:
    if not pause:
        ret, frame = vc.read()
        if frame is None:
            break



    # img = cv2.imread('Resources/lambo.png')
    img = frame
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos('hue min','TrackBars')
    h_max = cv2.getTrackbarPos('hue max', 'TrackBars')
    s_min = cv2.getTrackbarPos('sat min', 'TrackBars')
    s_max = cv2.getTrackbarPos('sat max', 'TrackBars')
    v_min = cv2.getTrackbarPos('val min', 'TrackBars')
    v_max = cv2.getTrackbarPos('val max', 'TrackBars')
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHSV,lower,upper)

    imgResult=cv2.bitwise_and(img,img,mask=mask)

    #cv2.imshow('oringin image',img)
    #cv2.imshow("mask image", mask)
    #cv2.imshow("result image", imgResult)
    imgStack = stackImages(0.6,([img,imgHSV],[mask,imgResult]))
    cv2.imshow('imgStack',imgStack)
 #   cv2.waitKey(0)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break
    elif key == 32:  # Press 'Space' to toggle pause
        pause = not pause

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(0)