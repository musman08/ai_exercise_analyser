import cv2
import numpy as np
import time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

count=0
direc=0
pTime=0

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

#For bicep curl
cap = cv2.VideoCapture('Videos/bicep_curl.mp4')
# cap = cv2.VideoCapture('vid/zaeem_indoor/bicep_curl.mp4')
# cap = cv2.VideoCapture('vid/usman_outdoor/bicep_curl.mp4')
# cap = cv2.VideoCapture('vid/usman_outdoor/bicep_curl.mp4')
# cap = cv2.VideoCapture('vid/talha/bicep_curl.mp4')




with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      cap.release()
      cv2.destroyAllWindows()
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image=cv2.resize(image, (1280, 720))
    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        landmarks = results.pose_landmarks.landmark
    except:
        pass
    
    #Getting Landmarks
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    
    
    #Calculating angles for bicep
    left_wrist_angle=calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_wrist_angle=calculate_angle(right_shoulder, right_elbow, right_wrist)
    #
    per=np.interp(left_wrist_angle, (50, 90), (0,100))
    bar=np.interp(left_wrist_angle, (50, 90), (0,100 ))
    # print(left_wrist_angle, per)
    if per==100:
        if direc==0:
            count+=0.5
            direc=1
    if per==0:
        if direc==1:
            count+=0.5
            direc=0
    print(count)
    
    # cv2.rectangle(image, (1100,100), (1175, 650), (0,255,0), 1)
    # cv2.rectangle(image, (1100,int(bar)), (1175, 650), (0,255,0), 3)
    cv2.putText(image, str(int(per)), (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), 4)

    
    
    
    cv2.rectangle(image, (0,450), (250, 720), (0,255,0), cv2.FILLED)
    cv2.putText(image, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255,0,0), 25)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(image, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
