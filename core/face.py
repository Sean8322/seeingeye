import os
import cv2
import face_recognition
import numpy
from tqdm import tqdm
import time
import csv
from time import sleep
from flask import Flask, render_template, Response

print ("Written by Sean Cheong")
#Please cd into directory of photos before running the program
print(time.asctime( time.localtime(time.time()) ))
face_names = []
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(1)


#success, image = video_capture.read()





name = ''
process_this_frame = True
image_encoding = []
photos = os.listdir('.')
photos.remove(".DS_Store")
print("Processing Photos")
for i in tqdm(range (0, 6)):
    image_encoding.append(face_recognition.face_encodings(face_recognition.load_image_file(photos[i]))[0])
    print (i, "photos processed")
    #print (image_encoding)


restarts = 0

check=[]
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # process every other frame of video to save time
    if process_this_frame:
        restarts = 0
        face_locations = face_recognition.face_locations(small_frame)
        unknown_encoding = face_recognition.face_encodings(small_frame, face_locations)
        if len(unknown_encoding) == 0:
            print("No Encoding")
        else:
            face_names = []
            totalMatches = 0
            tolerance = 0.5
            i=0
            name = "Unknown"
            while True:
                hold = numpy.array(image_encoding[i])
                #print(i)
                match = face_recognition.compare_faces([hold], unknown_encoding[0], tolerance=tolerance)
                if match[0]:
                    totalMatches += 1
                    name = photos[i]
                    #print(i)
                if i == 5:
                    if totalMatches < 1 and restarts<50:
                        i = 0
                        #sleep(0.05)
                        tolerance += 0.01
                        print("increasing tolerance")
                        print(tolerance)
                        totalMatches = 0
                        restarts+=1
                    elif totalMatches > 1 and restarts<50:
                        i = 0
                        #sleep(0.05)
                        tolerance -= 0.01
                        print("decreasing tolerance")
                        print(tolerance)
                        totalMatches = 0
                        restarts+=1
                    else:
                        face_names.append(name[:-4])
                        if name not in check:
                            check.append(name)
                            print (name[:-4] + ' Checked in on ' + time.asctime( time.localtime(time.time()) ))
                        break
                i+=1


    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

#    ret, jpeg = cv2.imencode('.jpg', frame)
    #print(jpeg)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release handle to the 
print("hi")
video_capture.release()
cv2.destroyAllWindows()
