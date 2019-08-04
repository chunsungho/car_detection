##### 가우시안블러 처리해서 차이점 시각화만 시켜놓은거
import cv2
import numpy as np

cap = cv2.VideoCapture("highway.mp4")  # highway

# video 가져와서 gray scale 처리 + 가우시안 블러처리 => 노이즈제거
_, first_frame = cap.read()

# /home/chun/PycharmProjects/opencv_project/main.py
# 파일경로설정을 제대로 안하면 여기서 에러가 날 수 있다.
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)



# frame을 하나하나 가져오는 while loop 를 돈다.
while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # core part
    # first frame 과 완전히 다른점 계산
    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)  # 두번째 인수가 노이즈 처리하는 threshold

    ##### 되나??
    _, contours, hierarchy = cv2.findContours(difference, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 500):
            x, y, w, h = cv2.boundingRect(contour)  # 이 함수 써서 좌표추출
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 얻어낸 좌표를 통해 빨간 박스처리
            cv2.imshow("difference", img)
    #            roi_vehchile = frame[y:y - 10 + h + 5, x:x - 8 + w + 10]

    # cv2.imshow("First frame", first_frame)
    # cv2.imshow("Frame", frame)
    # cv2.imshow("difference", difference)

    key = cv2.waitKey(20)  # cv2.waitKey()는 ESC를 누르면 27을 RETURN
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()

'''
### 다른점 찾아서 빨간박스 처리 하는거
import numpy as np
import cv2

video_file="highway.mp4"


kernel_dil = np.ones((20,20),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# 가우시안블러 처리가 들어가면 좋겠는데?

fgbg = cv2.createBackgroundSubtractorMOG2()        #MOG2 가 자동완성에 떴다. #background substraction 함수이다.
cap = cv2.VideoCapture(video_file)

while True:
    ret,frame = cap.read()
    fshape = frame.shape    #(세로,가로,채널수)    #채널수가 뭘 의미하는건지는 잘 모르겠음.
    # frame = frame[100:fshape[0]-100,:fshape[1]-100,:]       #cropping the video

    if ret == True:
        fgmask=fgbg.apply(frame)
        fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(fgmask,kernel_dil, iterations=1)
        _,contours,hierarchy=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>3000):
                x,y,w,h = cv2.boundingRect(contour)     #이 함수 써서 좌표추출
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)      #얻어낸 좌표를 통해 빨간 박스처리
                roi_vehchile = frame[y:y-10 + h+5,x:x-8+w+10]

        cv2.imshow("original",frame)

        if cv2.waitKey(1) == 27:
            break

    else:
        break


'''

'''



import cv2
import numpy as np

img=cv2.imread('chun.png',cv2.IMREAD_COLOR)
show_img=np.copy(img)

mouse_pressed = False
y = x = w = h = 0

labels = np.zeros(img.shape[:2],np.uint8)

#이미지에 네모박스 그리는 함수
def mouse_callback(event, _x, _y, flags, param):
    global show_img, x,y,w,h, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        x,y = _x, _y
        show_img = np.copy(img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            show_img = np.copy(img)
            cv2.rectangle(show_img, (x,y),
                          (_x, _y), (0, 255, 0), 3)

        elif event == cv2.EVENT_LBUTTONUP:
            mouse_pressed = False
            w, h = _x, -x, _y - y   #이게 뭔지 모르겠넹


label = cv2.GC_BGD
lbl_clrs = {cv2.GC_BGD: (0, 0, 0), cv2.GC_FGD: (255, 255, 255)}

def mouse_callback2(event, x, y, flags, param):
    global mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        cv2.circle(labels, (x, y), 5, label, cv2.FILLED)
        cv2.circle(show_img, (x, y), 5, lbl_clrs[label],
                   cv2.FILLED)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(labels, (x, y), 5, label, cv2.FILLED)
            cv2.circle(show_img, (x, y), 5, lbl_clrs[label],
                       cv2.FILLED)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback2)

while True:
    cv2.imshow('image', show_img)
    k = cv2.waitKey(1)          # k 는 뭐지 이건?

    if k == ord('a') and not mouse_pressed:
        break
    elif k == ord('l'):
        label = cv2.GC_FGD- label



cv2.destroyAllWindows()


# 4.


labels, bgdModel, fgdModel = cv2.grabCut(img, labels, (x,y,w,h), None, None, 5, cv2.GC_INIT_WITH_RECT)
#labels, bgdModel, fgdModel = cv2.grabCut(img, labels, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

show_img = np.copy(img)
show_img[(labels == cv2.GC_PR_BGD) |
         (labels == cv2.GC_BGD)] //= 3

cv2.imshow('image', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

# 5.




'''
