import numpy as np
import cv2
import math
import socket
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 5065

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

last = []

# Open Camera
try:
    default = 0 # Попробуйте изменить его на 1, если веб-камера не найдена
    capture = cv2.VideoCapture(default)
except:
    print("No Camera Source Found!")

while capture.isOpened():
    
    # Захват кадров с камеры
    ret, frame = capture.read()
    
   # Получить данные руки из прямоугольного подокна   
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    crop_image = frame[100:500, 100:500]
    
   # Применить размытие по Гауссу
    blur = cv2.GaussianBlur(crop_image, (3,3), 0)
    
   # Измените цветовое пространство с BGR -> HSV
   # hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
   # Создайте двоичное изображение, где белый будет цветом кожи, а остальные - черным
    mask2 = cv2.inRange(blur, np.array([2,0,0]), np.array([20,255,255]))
    
   # Ядро для морфологического преобразования  
    kernel = np.ones((5,5))
    
   # Применяем морфологические преобразования, чтобы отфильтровать фоновый шум
    dilation = cv2.dilate(mask2, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)    
       
   # Применить размытие по Гауссу и порог
    filtered = cv2.GaussianBlur(erosion, (3,3), 0)
    ret,thresh = cv2.threshold(filtered, 127, 255, 0)
    
   # Показать пороговое изображение
   # Найдите контуры
    contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    try:
        # Найдите контур с максимальной площадью
        contour = max(contours, key = lambda x: cv2.contourArea(x))
        
        # Создаем ограничивающий прямоугольник вокруг контура
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)
        
        # Найдите выпуклую оболочку
        hull = cv2.convexHull(contour)
        
       # Нарисуйте контур
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
        
       # Найдите дефекты выпуклости
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        
       # Используйте правило косинуса, чтобы найти угол дальней точки от начальной и конечной точки, то есть выпуклых точек (палец
       # подсказок) для всех дефектов
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            
        
            
           # если угол> 90 рисуем круг в дальней точке
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image,far,1,[0,0,255],-1)

            cv2.line(crop_image,start,end,[0,255,0],2)
            
            print("Defects : ", count_defects)

        # Показать необходимые изображения
        #cv2.imshow("Full Frame", frame)
        all_image = np.hstack((drawing, crop_image))
        cv2.imshow('Recognition', all_image)

        last.append(count_defects)
        if(len(last) > 5):
            last = last[-5:]
        
        # print(last)

        #Убедитесь, что рука раньше была широко открыта (3/4 пальца в предыдущих кадрах), а теперь сжата в кулак (0 пальцев)
        if(count_defects == 0 and 4 in last):
            last = []
            sock.sendto( ("JUMP!").encode(), (UDP_IP, UDP_PORT) )
            print("_"*10, "Jump Action Triggered!", "_"*10)
    except:
        pass

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()