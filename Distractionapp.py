import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import time
import threading

# Landmarks for Eyes

left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
left_iris = [474, 475, 476, 477]
right_iris = [469, 470, 471, 472]
left_top = [389]
left_bottom = [381]
right_top = [159]
right_bottom = [145]

# Initialize MediaPipe for face detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
elapsed_time = time.time()
start_time = None
total_distracted_time = 0.001
detection_active = False
video_thread = None

# Threshold Strictness Default Values
left_threshold = 2.95
right_threshold = 2.6
lblink_threshold = 0.8
rblink_threshold = 0.25

# Define slider variables as global
left_threshold_slider = None
right_threshold_slider = None
lblink_threshold_slider = None
rblink_threshold_slider = None

def distance(p1, p2):
    dist = sum([(i - j) ** 2 for i, j in zip(p1, p2)]) ** 0.5
    return dist

def position(center, right, left):
    center_right_dist = distance(center, right)
    total_distance = distance(right, left)
    threshold = center_right_dist / total_distance
    return threshold

# Function to update threshold labels while moving the sliders

def reset_threshold():
    global left_threshold, right_threshold, lblink_threshold, rblink_threshold
    left_threshold = 2.95
    right_threshold = 2.6
    lblink_threshold = 0.8
    rblink_threshold = 0.25
    left_threshold_slider.set(left_threshold)
    right_threshold_slider.set(right_threshold)
    lblink_threshold_slider.set(lblink_threshold)
    rblink_threshold_slider.set(rblink_threshold)

def start_detection():
    global cap, start_time, total_distracted_time, detection_active, video_thread, start_time, left_threshold, right_threshold, lblink_threshold, rblink_threshold

    # Move threshold updates here
    left_threshold = left_threshold_slider.get()
    right_threshold = right_threshold_slider.get()
    lblink_threshold = lblink_threshold_slider.get()
    rblink_threshold = rblink_threshold_slider.get()

    # Update threshold labels
    update_slider_labels(None)

    detection_active = True
    start_time = None
    total_distracted_time = 0.0
    video_thread = threading.Thread(target=video_capture)
    video_thread.start()

def video_capture():
    global detection_active, start_time, total_distracted_time, left_threshold, right_threshold, lblink_threshold, rblink_threshold

    with mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        curr_time = time.time()
        while detection_active:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[left_iris])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[right_iris])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv2.circle(frame, center_left, int(l_radius), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, center_right, int(r_radius), (255, 255, 255), 1, cv2.LINE_AA)

                threshold = position(center_right, mesh_points[left_eye[8]], mesh_points[left_eye[0]])
                left_blink = position(center_left, mesh_points[left_top[0]], mesh_points[left_bottom[0]])
                right_blink = position(center_right, mesh_points[right_top[0]], mesh_points[right_bottom[0]])

                if threshold <= float(right_threshold) or threshold >= float(left_threshold):
                    if start_time is None:
                        start_time = time.time()
                    cv2.putText(frame, 'Distracted', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif left_blink >= float(lblink_threshold) and right_blink <= float(rblink_threshold):
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 1:
                        cv2.putText(frame, 'Distracted', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    if start_time is not None:
                        end_time = time.time()
                        duration = end_time - start_time
                        total_distracted_time += duration
                        start_time = None
                if  threshold <= float(right_threshold) or threshold >= float(left_threshold) or (left_blink >= float(lblink_threshold) and right_blink <= float(rblink_threshold)):
                    cv2.putText(frame, f'Threshold {threshold:.2f} seconds', (180, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)
                else: 
                    cv2.putText(frame, f'Threshold {threshold:.2f} seconds', (180, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255, 0), 2)
                cv2.putText(frame, f'Total Distracted Time: {total_distracted_time:.2f} seconds', (120, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f'Total Elapsed Time: {time.time() - curr_time:.2f} seconds', (img_w - 290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('img', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        focused = ((time.time() - curr_time) - total_distracted_time) / (time.time() - curr_time) * 100
        print(f'Percent Time Focused: {focused:.2f}%')
        print(f'Percent Time Distracted: {100 - focused:.2f}%')
        print(f'Total Elapsed time: {time.time() - curr_time:.2f} seconds')

def stop_detection():
    global detection_active, video_thread

    detection_active = False
    if video_thread is not None:
        video_thread.join()


root = tk.Tk()
root.title("Distracted Driver Detection")

left_threshold_var = tk.StringVar()
right_threshold_var = tk.StringVar()
lblink_threshold_var = tk.StringVar()
rblink_threshold_var = tk.StringVar()

def update_slider_labels(event=None):
    left_threshold_label.config(text=f'Left Threshold: {left_threshold_slider.get():.2f}')
    right_threshold_label.config(text=f'Right Threshold: {right_threshold_slider.get():.2f}')
    lblink_threshold_label.config(text=f'Left Blink Threshold: {lblink_threshold_slider.get():.2f}')
    rblink_threshold_label.config(text=f'Right Blink Threshold: {rblink_threshold_slider.get():.2f}')

# Create labels to display threshold values beside the sliders
left_threshold_label = ttk.Label(root, text=f'Left Threshold: {left_threshold:.2f}')
left_threshold_label.pack()
left_threshold_slider = ttk.Scale(root, from_=0, to=4, length=200, orient="horizontal", variable=left_threshold_var)
left_threshold_slider.set(left_threshold)
left_threshold_slider.bind('<Motion>', update_slider_labels)  # Bind motion event to update labels
left_threshold_slider.pack()

right_threshold_label = ttk.Label(root, text=f'Right Threshold: {right_threshold:.2f}')
right_threshold_label.pack()
right_threshold_slider = ttk.Scale(root, from_=0, to=4, length=200, orient="horizontal", variable=right_threshold_var)
right_threshold_slider.set(right_threshold)
right_threshold_slider.bind('<Motion>', update_slider_labels) 
right_threshold_slider.pack()

lblink_threshold_label = ttk.Label(root, text=f'Left Blink Threshold: {lblink_threshold:.2f}')
lblink_threshold_label.pack()
lblink_threshold_slider = ttk.Scale(root, from_=0, to=4, length=200, orient="horizontal", variable=lblink_threshold_var)
lblink_threshold_slider.set(lblink_threshold)
lblink_threshold_slider.bind('<Motion>', update_slider_labels) 
lblink_threshold_slider.pack()

rblink_threshold_label = ttk.Label(root, text=f'Right Blink Threshold: {rblink_threshold:.2f}')
rblink_threshold_label.pack()
rblink_threshold_slider = ttk.Scale(root, from_=0, to=4, length=200, orient="horizontal", variable=rblink_threshold_var)
rblink_threshold_slider.set(rblink_threshold)
rblink_threshold_slider.bind('<Motion>', update_slider_labels)
rblink_threshold_slider.pack()

start_button = ttk.Button(root, text="Start Detection", command=start_detection)
stop_button = ttk.Button(root, text="Stop Detection", command=stop_detection)
reset_button = ttk.Button(root, text="Reset Thresholds", command=reset_threshold)

start_button.pack()
stop_button.pack()
reset_button.pack()

# Initialize threshold labels with their initial values
left_threshold_var.set(f'Left Threshold: {left_threshold:.2f}')
right_threshold_var.set(f'Right Threshold: {right_threshold:.2f}')
lblink_threshold_var.set(f'Left Blink Threshold: {lblink_threshold:.2f}')
rblink_threshold_var.set(f'Right Blink Threshold: {rblink_threshold:.2f}')

root.mainloop()