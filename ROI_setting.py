import cv2
import os


image_folder = '/home/chun/PycharmProjects/openCV_2/images'


mouse_is_pressing = False
start_x, starty = -1, -1
flag = 0

def mouse_callback(event,x,y,flags,param):
    global start_x, start_y,mouse_is_pressing, flag

    img_result = img_color.copy()

    if event == cv2.EVENT_LBUTTONDOWN:

        mouse_is_pressing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:

        if mouse_is_pressing:
            if start_x < x :
                cv2.rectangle(img_result, (start_x, start_y), (x, y), (0, 255, 0), 3)
            elif x < start_x :
                flag = 1
                cv2.rectangle(img_result, (x, y), (start_x, start_y), (0, 255, 0), 3)

        cv2.imshow("img_color", img_result)

    elif event == cv2.EVENT_LBUTTONUP:

        mouse_is_pressing = False
        cv2.rectangle(img_result, (start_x, start_y), (x, y), (0, 255, 0), 3)
        cv2.imshow("img_color", img_result)

        # 원본 영역에서 두 점 (start_y, start_x), (x,y)로 구성되는 사각영역을 잘라내어 변수 img_cat이 참조하도록 합니다.
        if flag == 0 :
            img = img_color[ start_y:y, start_x:x]

        elif flag == 1:
            img = img_color[y:start_y, x:start_x]
        cv2.imshow("img_color1", img)


## main ##

for n, image_file in enumerate(os.scandir(image_folder)):
    img_color = cv2.imread(image_file.path)
    cv2.imshow("img_color", img_color)
    cv2.setMouseCallback('img_color', mouse_callback)

    key = cv2.waitKey(0) & 0xff
    if key == ord('q'):
        # 지정된 경로에 저장
        continue
    else :
        break






    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #ax.imshow(image)
    #plt.show()


#img_color = cv2.imread('giraffe.jpg', cv2.IMREAD_COLOR )
#cv2.imshow("img_color", img_color)
#



#cv2.destroyAllWindows()

# 해야할 일 : 지정된 경로에서 이미지를 순차적으로 가져와서 캡쳐하는거는 또 다른 폴더에다가 차곡차곡 저장하는 알고리즘

