import cv2

# 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
vidcap = cv2.VideoCapture('vaseline3.mp4')    # 이 부분 바꿔서 동영상 파일을 가져와서 캡쳐하기가 가능.

count = 0
frame_skip = 0

while (vidcap.isOpened()):

    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    ret, image = vidcap.read()
    frame_skip += 1
    if(frame_skip % 10 == 0):
        # 캡쳐된 이미지를 저장하는 함수
        cv2.imwrite("vaseline_rawData/3frame%d.jpg" % count, image)

        print('Saved 3frame%d.jpg' % count)
        count += 1

vidcap.release()
