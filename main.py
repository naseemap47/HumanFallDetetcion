import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

img = cv2.imread('t.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = pose.process(img_rgb)

if result.pose_landmarks:
    mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

cv2.imshow('img', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
