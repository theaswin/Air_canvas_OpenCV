import cv2
import numpy as np
import HandTrackingModule as htm


cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

detector = htm.handDetector()
drawing_color = (0,255,0)


# Creating canvas for drawing
image_canvas = np.zeros((820,1280,3),np.uint8)

brush_size = 5 # Default brush size 


while True:
#1.import image
    success,image = cap.read()
    image = cv2.resize(image,(1280,820))
    image = cv2.flip(image,1)
    cv2.rectangle(image,(0,0),(1280,110),(0,0,0),cv2.FILLED) # background box

    cv2.rectangle(image,(10,10),(210,100),(0,0,255),cv2.FILLED) # red
    cv2.rectangle(image,(230,10),(430,100),(0,255,0),cv2.FILLED) # green
    cv2.rectangle(image,(450,10),(650,100),(255,0,0),cv2.FILLED) # blue
    cv2.rectangle(image,(670,10),(870,100),(0,255,255),cv2.FILLED) # yellow
    cv2.rectangle(image,(890,10),(1270,100),(255,255,255),cv2.FILLED) # eraser
    cv2.putText(image,'Eraser',(1035,65),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)


# Find hand landmarks
    image = detector.findHands(image)
    lmlist = detector.findPosition(image) # landmarks of detected list

    if len(lmlist) != 0:
        x1,y1 = lmlist[8][1:] # values of 8th landmark except the first values [handpoint,x-cordinate,y-cordinate] index finger
        x2,y2 = lmlist[12][1:] # values of 12th landmark except the first values [handpoint,x-cordinate,y-cordinate] near finger index

# Check which finger is up
    fingers =detector.fingersUp()


# Selection mode
    if fingers[1] and fingers[2]:
        print('selection model')

        # this is the point after selecting a colour , The point initially set as zero
        xp,yp=0,0

        if y1<120:
            if 10<x1<230:
                drawing_color = (0,0,255)
                

            elif 250<x1<470:
                drawing_color = (0,255,0)
                

            elif 490<x1<710:
                drawing_color = (255,0,0)

            elif 720<x1<880:
                drawing_color = (0, 255, 255)

            
            elif 890<x1<1270:
                drawing_color=(0,0,0)
        
        # Showing rectangle tip of  fingers
        cv2.rectangle(image,(x1,y1),(x2,y2),drawing_color,cv2.FILLED)
# Showing circle tip of the index finger
    if (fingers[1] and not fingers[2]):
        cv2.circle(image,(x1,y1),15,drawing_color,1,cv2.FILLED)

        if xp ==0 and yp==0:
            xp=x1
            yp=y1
            # Drawing line
        if drawing_color ==(0,0,0):
            cv2.line(image,(xp,yp),(x1,y1),drawing_color,thickness=20)
            cv2.line(image_canvas,(xp,yp),(x1,y1),drawing_color,thickness=20)
        else:
            cv2.line(image,(xp,yp),(x1,y1),drawing_color,brush_size)
            cv2.line(image_canvas,(xp,yp),(x1,y1),drawing_color,brush_size)
        xp,yp = x1,y1




    imgGray = cv2.cvtColor (image_canvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,20,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

    image = cv2.bitwise_and(image,imgInv)
    image  = cv2.bitwise_or(image,image_canvas)


    image = cv2.addWeighted(image,1,image_canvas,0.5,0)



    cv2.imshow('virtual painter',image)


    if cv2.waitKey(1) & 0xFF == 27:
        break