import cv2
import time
import numpy as np           #numerical computing library
import hand_detector as hd   #it is a custom module
import pyautogui             #library for automating keyboard and mouse movement
import webcolors             #library to convert rgb to color names

def get_color_name(rgb_tuple):
    try:
        color_name = webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        color_name = None

    return color_name

wCam, hCam = 640, 480     # width and height of the camera capture window
frameR = 100              #size of frame around the camera capture window where hand detection wont occur
smoothening = 7           #smoothening factor of the mouse movement, no of frames
scroll_speed=10           #the speed of scroll program

pTime = 0                 #used to keep track of previous time
plocX, plocY = 0, 0       #used to keep track of previous loction of mouse cursor
clocX, clocY = 0, 0       #used to store the current location of the cursor

cap = cv2.VideoCapture(0) #this is a method used to initialize the camera capture
cap.set(3, wCam)          #sets the height of the screen based on the size of monitor
cap.set(4, hCam)
detector = hd.handDetector(detectionCon=0.7)
wScr, hScr = pyautogui.size()
print(wScr, hScr)


while True:
    success, img = cap.read()         #reads the camera capture
    img = detector.findHands(img)     #finds hand positions
    lmList, bbox = detector.findPosition(img) #find positions of landmarks of hands
    output = img.copy()               #creates a copy of the frame

    if len(lmList) != 0:              #coordinates are calculated
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[16][1:]

        fingers = detector.fingersUp()  #uses methpod from other class to detect which fingers are extended

        #draw a rectangle around the region where hand is detected
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (205, 250, 255), -1)
        img = cv2.addWeighted(img, 0.5, output, 1 - .5, 0, output)

        # Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move Mouse
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 6, (255, 28, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 6, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

        # Three fingers are up: Scrolling Mode up
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            if fingers[4]==0:
                length, img, lineInfo = detector.findDistance(12, 16, img)
                pyautogui.scroll(int(length / 2))
            else:
                # Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, 255))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, 255))

                # Get the RGB values of the pixel under the fingertip
                pixel_color = img[int(y1), int(x1)]
                r, g, b = pixel_color[2], pixel_color[1], pixel_color[0]

                # Find the closest color name for the detected RGB values
                color_name = webcolors.rgb_to_name((r, g, b))
                print("Detected color: ", color_name)
                time.sleep(3)



        # Three fingers are up: Scrolling mode down
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4]==1:
            length, img, lineInfo = detector.findDistance(12, 16, img)
            scroll_speed=-int(length/2)
            pyautogui.scroll(scroll_speed)

        # Index and little finger up: Take screenshot
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4]==1:
            img = pyautogui.screenshot()
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("screenshot.png", img)
            time.sleep(5)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow("Virtual mouse monitor", cv2.flip(img, 1))
    cv2.setWindowProperty("Virtual mouse monitor", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)











