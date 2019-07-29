import cv2
import numpy as np

cap = cv2.VideoCapture("highway.mp4")

# video 가져와서 gray scale 처리 + 가우시안 블러처리 => 노이즈제거
_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

# ..
while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

# core part
# first frame 과 완전히 다른점 계산
    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    cv2.imshow("First frame", first_frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("difference", difference)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


'''
import numpy as np
import cv2

video_file="highway.mp4"


kernel_dil = np.ones((20,20),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()        #MOG2 가 자동완성에 떴다. #background substraction 함수이다.
cap = cv2.VideoCapture(video_file)

#cv2.imshow("video_capture",cap)

while True:
    ret,frame = cap.read()
    fshape = frame.shape
    frame = frame[100:fshape[0]-100,:fshape[1]-100,:]       #cropping the video

    if ret == True:
        fgmask=fgbg.apply(frame)
        fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(fgmask,kernel_dil, iterations=1)
        (contours,hierarchy)=cv2.findContours\
            (dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>3500):
                x,y,w,h = cv2.boundingRect(contour)
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                roi_vehchile = frame[y:y-10 + h+5,x:x-8+w+10]

        cv2.imshow("original",frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    else:
        break


'''
