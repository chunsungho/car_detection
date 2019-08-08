
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
