import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh() 

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    H, W = frame.shape[:2]    
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    face_mesh_out = face_mesh.process(rgb)
    landmark_det = face_mesh_out.multi_face_landmarks

    if landmark_det :
        landmark_cors = landmark_det[0].landmark
        for cor in landmark_cors:
            cor_frame_y = cor.y * H 
            cor_frame_x = cor.x * W 
            frame = cv2.circle(frame , (int(cor_frame_x),int(cor_frame_y)),1,(0,255,0),1)

    cv2.imshow('Capture Frame',frame)
    key = cv2.waitKey(30)
    if key == 115:
        break