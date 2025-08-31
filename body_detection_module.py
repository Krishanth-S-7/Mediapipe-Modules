import cv2 as cv
import mediapipe as mp
import time
import numpy as np

class BodyDetector: 
    def __init__(self,mode=False,mod_complex=1,smooth_landmarks=True,enable_segment=False,smooth_segment=True,min_det_con=0.5,min_track_con=0.5):
        self.mode = mode
        self.mod_complex = mod_complex
        self.smooth_landmarks = smooth_landmarks
        self.enable_segment = enable_segment
        self.smooth_segment = smooth_segment
        self.min_det_con = min_det_con
        self.min_track_con = min_track_con
        self.mppose = mp.solutions.pose
        self.pose = self.mppose.Pose(self.mode,self.mod_complex,self.smooth_landmarks,self.enable_segment,self.smooth_segment,self.min_det_con,self.min_track_con)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findbodies(self,img,draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)
    def getcoords(self,img,draw=False):
        self.lmlist = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx, cy), 30, 255, cv.FILLED)
        return self.lmlist
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
    def show_fps(self,img,ctime,ptime):
        fps = 1 / (ctime - ptime)
        cv.putText(img, str(int(fps)), (30, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)




def main():
    cap = cv.VideoCapture(0)
    ctime = 0
    ptime = 0
    detector = BodyDetector()
    while True:
        ret, img = cap.read()
        if ret:
            detector.findbodies(img)
            coords = detector.getcoords(img)
            if coords:
                k = detector.getangle(12,14,16)
                cv.putText(img, str(int(k)), (coords[14][1]-30, coords[14][2]+30), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
            ctime = time.time()
            detector.show_fps(img,ctime,ptime)
            ptime = ctime
            cv.imshow('hello', img)
            cv.waitKey(1)
if __name__ == '__main__':
    main()