import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pygame

# Landmarks for Eyes
left_eye = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
right_eye = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
left_iris = [474, 475, 476, 477]
right_iris = [469, 470, 471, 472]
left_top = [386]
left_bottom = [374]
right_top = [159]
right_bottom = [145]

#MediaPipe's face detection
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
total_distracted_time = 0.0
detection_active = False
video_thread = None

# Threshold Strictness Default Values - Can change if you want
left_threshold = 2.95
right_threshold = 2.6
lblink_threshold = 0.8
rblink_threshold = 0.25

pygame.mixer.init()

# Global variables for audio playback - Way too many errors but works fine now
audio_playing = False
audio_file = "test.mp3"  # Replace with your audio file path

def distance(p1, p2):
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist

def position(center, point1, point2):
    center_point1_dist = distance(center, point1)
    total_distance = distance(point1, point2)
    if total_distance == 0:
        return 0
    threshold = center_point1_dist / total_distance
    return threshold

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
    global cap, total_distracted_time, detection_active, video_thread, left_threshold, right_threshold, lblink_threshold, rblink_threshold

    # Update thresholds from sliders
    left_threshold = float(left_threshold_slider.get())
    right_threshold = float(right_threshold_slider.get())
    lblink_threshold = float(lblink_threshold_slider.get())
    rblink_threshold = float(rblink_threshold_slider.get())

    # Update threshold labels
    update_slider_labels()

    detection_active = True
    total_distracted_time = 0.0
    video_thread = threading.Thread(target=video_capture)
    video_thread.start()

def video_capture():
    global detection_active, total_distracted_time, audio_playing
    with mp_face_mesh.FaceMesh(refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        curr_time = time.time()
        is_distracted = False
        start_time = None
        while detection_active:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points = np.array(
                    [np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                     for p in results.multi_face_landmarks[0].landmark])

                # Compute thresholds and positions
                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(
                    mesh_points[left_iris])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(
                    mesh_points[right_iris])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                threshold = position(
                    center_right, mesh_points[left_eye[8]], mesh_points[left_eye[0]])
                left_blink = position(
                    mesh_points[left_top[0]], mesh_points[left_bottom[0]], mesh_points[left_eye[0]])
                right_blink = position(
                    mesh_points[right_top[0]], mesh_points[right_bottom[0]], mesh_points[right_eye[0]])

                # Determine if distracted
                current_distracted = False
                if threshold <= right_threshold or threshold >= left_threshold:
                    current_distracted = True
                    cv2.putText(frame, 'Distracted', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if left_blink >= lblink_threshold and right_blink <= rblink_threshold:
                    current_distracted = True
                    cv2.putText(frame, 'Distracted', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Handle state transitions
                if current_distracted != is_distracted:
                    if current_distracted:
                        # Just became distracted
                        start_time = time.time()
                        # Start playing audio if not already playing
                        if not audio_playing:
                            pygame.mixer.music.load(audio_file)
                            pygame.mixer.music.play(-1)  # Loop indefinitely
                            audio_playing = True
                    else:
                        # Individual is not distracted anymore
                        if start_time is not None:
                            duration = time.time() - start_time
                            total_distracted_time += duration
                            start_time = None
                        # Stop playing audio if it is playing 
                        if audio_playing:
                            pygame.mixer.music.stop()
                            audio_playing = False
                    is_distracted = current_distracted

                # Display information for individual to monitor
                color = (0, 0, 255) if current_distracted else (0, 255, 0)
                cv2.putText(frame, f'Threshold {threshold:.2f}', (180, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f'Total Distracted Time: {total_distracted_time:.2f}s', (120, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f'Total Elapsed Time: {time.time() - curr_time:.2f}s',
                            (img_w - 290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('img', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # If still distracted at the end, update the total time
        if is_distracted and start_time is not None:
            duration = time.time() - start_time
            total_distracted_time += duration

        total_time = time.time() - curr_time
        focused = ((total_time) - total_distracted_time) / (total_time) * 100
        print(f'Percent Time Focused: {focused:.2f}%')
        print(f'Percent Time Distracted: {100 - focused:.2f}%')
        print(f'Total Elapsed time: {total_time:.2f} seconds')
        cv2.destroyAllWindows()

        # Ensure audio stops when detection ends
        if audio_playing:
            pygame.mixer.music.stop()

def stop_detection():
    global detection_active, video_thread

    detection_active = False
    if video_thread is not None:
        video_thread.join()

root = tk.Tk()
root.title("Distracted Student Detection")

def update_slider_labels(event=None):
    left_threshold_label.config(
        text=f'Left Threshold: {float(left_threshold_slider.get()):.2f}')
    right_threshold_label.config(
        text=f'Right Threshold: {float(right_threshold_slider.get()):.2f}')
    lblink_threshold_label.config(
        text=f'Left Blink Threshold: {float(lblink_threshold_slider.get()):.2f}')
    rblink_threshold_label.config(
        text=f'Right Blink Threshold: {float(rblink_threshold_slider.get()):.2f}')

# Create labels to display threshold values beside the sliders
left_threshold_label = ttk.Label(
    root, text=f'Left Threshold: {left_threshold:.2f}')
left_threshold_label.pack()
left_threshold_slider = ttk.Scale(
    root, from_=0, to=4, length=200, orient="horizontal")
left_threshold_slider.set(left_threshold)
left_threshold_slider.bind('<Motion>', update_slider_labels)
left_threshold_slider.pack()

right_threshold_label = ttk.Label(
    root, text=f'Right Threshold: {right_threshold:.2f}')
right_threshold_label.pack()
right_threshold_slider = ttk.Scale(
    root, from_=0, to=4, length=200, orient="horizontal")
right_threshold_slider.set(right_threshold)
right_threshold_slider.bind('<Motion>', update_slider_labels)
right_threshold_slider.pack()

lblink_threshold_label = ttk.Label(
    root, text=f'Left Blink Threshold: {lblink_threshold:.2f}')
lblink_threshold_label.pack()
lblink_threshold_slider = ttk.Scale(
    root, from_=0, to=4, length=200, orient="horizontal")
lblink_threshold_slider.set(lblink_threshold)
lblink_threshold_slider.bind('<Motion>', update_slider_labels)
lblink_threshold_slider.pack()

rblink_threshold_label = ttk.Label(
    root, text=f'Right Blink Threshold: {rblink_threshold:.2f}')
rblink_threshold_label.pack()
rblink_threshold_slider = ttk.Scale(
    root, from_=0, to=4, length=200, orient="horizontal")
rblink_threshold_slider.set(rblink_threshold)
rblink_threshold_slider.bind('<Motion>', update_slider_labels)
rblink_threshold_slider.pack()

start_button = ttk.Button(root, text="Start Detection", command=start_detection)
stop_button = ttk.Button(root, text="Stop Detection", command=stop_detection)
reset_button = ttk.Button(root, text="Reset Thresholds", command=reset_threshold)

#Make sure all buttons work
start_button.pack()
stop_button.pack()
reset_button.pack()
root.mainloop()
pygame.mixer.quit()
