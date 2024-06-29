# Git-DDD
for explain the project about Drowsiness-Detections-Systems

## main.py
import os

from scipy.spatial import distance
from imutils import face_utils
import dlib
import cv2
import socket

os.chdir(r"C:\\Users\\A.N.A.S.N.H\\Desktop\\Nourhan\\DrowsyFaceLandmarks\\#LastElLast#\\car\\venv")
model_path = "C:\\Users\\A.N.A.S.N.H\\Desktop\\Nourhan\\DrowsyFaceLandmarks\\shape_predictor_68_face_landmarks.dat"

esp32_ip = '192.168.1.38'
esp32_port = 80

MOTOR1_FORWARD = "MOTOR1_FORWARD"
MOTOR2_FORWARD = "MOTOR2_FORWARD"
STOP_MOTORS = "STOP_MOTORS"
FlucherON = "FlucherON"
BuzzerON = "BuzzerON"
BuzzerOFF = "BuzzerOFF"
FlucherOFF = "FlucherOFF"
Clear_LCD = "Clear_LCD"
Send_Message="Send_Message"

def send_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((esp32_ip, esp32_port))
        s.sendall(f"{command}\n".encode())
        print(f"تم إرسال الأمر: {command}")

def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear

send_command(MOTOR1_FORWARD)
send_command(MOTOR2_FORWARD)

def detect():
    counter = 0
    thresh = 0.25
    flag = 0

    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor(model_path)  # Dat file is the crux of the code

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    cap = cv2.VideoCapture("http://192.168.1.37:8080//video")  #cap = cv2.VideoCapture(0)
    address="http://192.168.1.37:8080//video"
    cap.open(address)

    while True:
        ret, frame =cap.read()
        height =640
        width = 640
        dsize = (width, height)
        frame = cv2.resize(frame, dsize, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        subjects = detect(gray, 0)
        num_faces=len(subjects)

        if num_faces == 0:
            flag += 1
            if flag == 5:
               send_command(BuzzerON)
            if flag >= 20 and flag < 40:
               if flag == 20:
                  send_command("PRINT:     Drowsy     ")
               send_command(BuzzerON)
            if flag >= 40 and flag < 50 :
               if flag % 2 == 0 :
                  send_command(FlucherON)
               else:
                  send_command(FlucherOFF)
            if flag == 50 :
                counter +=1
                if counter == 1 :
                   send_command(STOP_MOTORS)
            if flag > 50 :
               if flag % 2 == 0 :
                  send_command(FlucherON)
               else :
                  send_command(FlucherOFF)

        else :
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                (x, y, w, h) = cv2.boundingRect(shape)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < thresh :
                   flag += 1
                   if flag >= 20 and flag < 40:
                      if flag == 20 :
                         send_command("PRINT:     Drowsy     ")
                      send_command(BuzzerON)
                   if flag >= 40 and flag < 50 :
                      if flag % 2 == 0 :
                         send_command(FlucherON)
                      else :
                         send_command(FlucherOFF)
                   if flag == 50 :
                      counter += 1
                      if counter == 1 :
                         send_command(STOP_MOTORS)
                   if flag > 50 :
                      if flag % 2 == 0:
                         send_command(FlucherON)
                      else:
                         send_command(FlucherOFF)

                else :
                   flag = 0
                   send_command(Clear_LCD)
                   send_command(BuzzerOFF)
                   send_command(FlucherOFF)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
           break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__' :
   detect()
