import mediapipe as mp
import cv2 as cv
import time
import numpy as np



class HandDetector():
    def __init__(self,static_image=False,number_hands=2,model_complexity=1,det_con=0.5,trac_con=0.5):
        self.static_image = static_image
        self.number_hands = number_hands
        self.model_complexity = model_complexity
        self.det_con = det_con
        self.trac_con = trac_con
        self.result = None

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image,self.number_hands,self.model_complexity,self.det_con,self.trac_con)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self,img,draw=True):
        rgbimg = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.result = self.hands.process(rgbimg)
        if self.result.multi_hand_landmarks:
            for eachhand in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, eachhand, self.mpHands.HAND_CONNECTIONS)
    def getcoord(self,img,handno=0,draw=False):
        self.lmlist = []
        if self.result and self.result.multi_hand_landmarks:
            try:
                ourhand=self.result.multi_hand_landmarks[handno]
            except:
                return []
            for id, lm in enumerate(ourhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 30, 255, cv.FILLED)
        return self.lmlist
    def show_fps(self,img,ctime,ptime):
        fps = 1 / (ctime - ptime)
        cv.putText(img, str(int(fps)), (30, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    def getangle(self,p1,p2,p3):
        k,x1,y1 = self.lmlist[p1]
        k,x2,y2 = self.lmlist[p2]
        k,x3,y3 = self.lmlist[p3]
        v1 = np.array([x1 - x2, y1 - y2])
        v2 = np.array([x3 - x2, y3 - y2])

        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
        angle_deg = np.degrees(angle_rad)

        return int(angle_deg)


def main():
    cap = cv.VideoCapture(0)
    detector = HandDetector()
    ctime = 0
    ptime = 0
    while True:
        success, img = cap.read()
        ctime = time.time()
        detector.show_fps(img, ctime, ptime)
        # fps = 1 / (ctime - ptime)
        ptime = ctime
        # cv.putText(img, str(int(fps)), (30, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        detector.findhands(img)
        coords = detector.getcoord(img)
        if coords:
            print(detector.getangle(5,6,7))
        cv.imshow('Video', img)
        cv.waitKey(1)
if __name__ == '__main__':
    main()