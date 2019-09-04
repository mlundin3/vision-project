import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

person_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
tracker = cv2.TrackerMedianFlow_create()
degrees = 0
distance = '0'
x = 640
y = 480

def detect_person(img):

    global tracker
    
    person_img = img.copy()
  
    person_rects = person_cascade.detectMultiScale(person_img,scaleFactor=1.3, minNeighbors=3)

    if person_rects == ():
        print('No person found!')
    else:
        print('Person found!')
        print(person_rects)
        for (x,y,w,h) in person_rects:
            person_rects = (x,y,w,h)
        tracker = cv2.TrackerMedianFlow_create()
        tracker.init(person_img, person_rects)


def calc_degrees(x):
    return round((x/640)*60 - 30, 1)

def calc_distance(w):
    if w < 60:
        return '>2'
    if w > 150:
        return '<0.5'
    else:
        if w >= 90:
            return str(round(1 - ((w-90)/60)*0.5,2))
        if w < 90:
            return str(round(1 + ((90-w)/30)*1,2))

cap = cv2.VideoCapture(0)

num_frames = 0

while True:
    
    ret, frame = cap.read()

    cv2.putText(frame, 'Degrees: ' + str(degrees), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, 'Distance: ' + distance + 'm', (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    if num_frames < 100:
        cv2.putText(frame, 'INITALIZING', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)

    if len(str(num_frames)) > 2 and str(num_frames)[-2:] == '00':
        detect_person(frame)

    if num_frames > 100:
        success, roi = tracker.update(frame)
        (x, y, w, h) = tuple(map(int, roi))
        if success:
            # Tracking success
            degrees = calc_degrees(x+(w/2))
            distance = calc_distance(w)
            p1 = (x, y)
            p2 = (x + w, y + h)
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
        else:
            detect_person(frame)

    cv2.imshow('img',frame)

    num_frames += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()