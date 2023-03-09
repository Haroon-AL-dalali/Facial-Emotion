import os
from turtle import position
import cv2
import numpy as np
from keras.models import load_model
import face_recognition
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
import mysql.connector as mc
import datetime as dt
import time
model=load_model('model_file_30epochs.h5')

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
path = 'photos'
images=[]
classNames = []
personsList = os.listdir(path)


for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(image):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
status = [0,0,0,0,0,0,0]


class Feelings(QDialog):
   
    def __init__(self):
        super(Feelings,self).__init__()
        loadUi("Feelings.ui",self)
        self.btnTake.clicked.connect(self.Take)
    
    def Take(self):
        cap = cv2.VideoCapture(0)
        while True:
            _,frame=cap.read()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            faces= faceDetect.detectMultiScale(gray, 1.3, 3)
            for x,y,w,h in faces:
                sub_face_img=gray[y:y+h, x:x+w]
                resized=cv2.resize(sub_face_img,(48,48))
                faceCurrentFrame = face_recognition.face_locations(resized)
                encodeCurrentFrame = face_recognition.face_encodings(resized,faceCurrentFrame)

                for encodeface,faceLoc in zip(encodeCurrentFrame,faceCurrentFrame):
                    matches = face_recognition.compare_faces(encodeListKnown,encodeface)
                    faceDis = face_recognition.face_distance(encodeListKnown,encodeface)

                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        dateTimes = str(dt.datetime.now())[:19]
                        normalize=resized/255.0
                        reshaped=np.resize(normalize, (1, 48, 48, 1))
                        result=model.predict(reshaped)
                        label=np.argmax(result, axis=1)[0]
                        status[label]+=1
                        print(status)
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
                        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
                        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                        for i in status:
                            if i > 9:
                                index= status.index(i)
                                print(f"the index is {index}")
                                mydb = mc.connect(
                                host = "localhost",
                                user = "root",
                                password = "",
                                database = "facialEmotion"
                                )
                                mycursor2 = mydb.cursor()
                                query = "INSERT INTO `users`( `name`, `emotion`,`dateCreation`) VALUES ('"+name+"', '"+labels_dict[index]+"','"+dateTimes+"')"
                                mycursor2.execute(query)
                                mydb.commit()
                           
                                cv2.destroyAllWindows()
                                return

            cv2.imshow("face emotion",frame)
            cv2.waitKey(1)                          


app = QApplication(sys.argv)
mainwindow=Feelings()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(500)
widget.setFixedHeight(450)
widget.show()
app.exec_()