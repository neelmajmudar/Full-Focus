# 📌 Full Focus

A desktop app that tracks and enhances your focus using real-time distraction monitoring powered by eye and blink detection.

---

## About

Full Focus is a light and intuitive productivity application that helps users stay focused during work or study sessions. Leveraging real-time tracking of eye position and blink frequency, the app identifies distraction states and gently nudges you back on track with customizable alerts.

- **Live version**: Fully-featured and accessible at **[fullfocus.vercel.app](https://fullfocus.vercel.app/)**.
- **Beta on GitHub**: The version here showcases a simplified architecture, ideal for those learning how the app was built—from tracking to visualization—without the full UI polish.

This was a really fun project I did during the Summer of 2023. While I would do my work, I kept noticing that I would easily get distracted as I'd constantly reach for my phone. So, to counteract this, I made an OpenCV project that would help me stay focused. I looked through Google's documentation on Mediapipe and several YouTube tutorials online to figure out how to draw a mesh grid of my face on OpenCV. After figuring out the coordinates of the points drawn on my face, I reduced them to just my eyes and used a simple distance formula to keep track of the distance from the center of my pupil to each end of my eye. With this, I was able to set a threshold that would alert the user when they are looking anywhere outside the range of their computer screen from up, down, left, right, and closed-eye movements. I used tkinter to create a simple, easy-to-use GUI for this program and allowed users to change various thresholds so it could be more suited to their setup/needs. As an addition, each time the user was distracted, I even added a function to play audio in the background to remind the user to get back to work. Once the user quits the application, they can even see various metrics like the total time elapsed, time distracted, percent time focused, etc.

Demo video is too big to post here so please take a look at it here: https://imgur.com/T4zbWTq

---

## Features

- **Real-time distraction detection**  
  Tracks eye movements and blink rates via webcam to determine when you lose focus.

- **Customizable alerts**  
  Set threshold values, delay times, and alert repetitions to tailor feedback to your preferences.  
  :contentReference[oaicite:2]{index=2}

- **Focus session metrics**  
  Gain insights such as “Total Time Focused,” “Time Distracted,” and session-level summaries.  
  :contentReference[oaicite:3]{index=3}

- **Multi-platform support**  
  Available on Windows and macOS (Mac version launched as of 5 months ago).  
  :contentReference[oaicite:4]{index=4}

- **Gentle audio notifications**  
  Choose or upload alert sounds to bring your attention back without disrupting your flow.

---

## Project Overview

Explore how the simplified beta version works:

### 1. **Eye & Blink Tracking**  
Webcam feed is processed frame-by-frame to detect pupil positions and blink rate, allowing distraction detection logic.

### 2. **Distraction Logic**  
Configurable thresholds compare eye positions and blink frequencies to detect attention drift.

### 3. **Alert System**  
Trigger audio feedback once distraction crosses thresholds, repeated at set intervals.

### 4. **Session Analytics**  
Logs and aggregates focus versus distraction durations; provides a session report.

### 5. **UI (Beta on GitHub)**  
Lightweight React-based interface to display live metrics and session summaries, without the aesthetics or polish of the live version.

---

## Why Both Versions?

- **Live version (fullfocus.vercel.app)**: The complete, polished experience for end-users—custom-built UI, responsive alerts, full settings panel, and session visualization.
- **GitHub beta**: The educational build—minimal UI, core distraction detection logic, and session logging. Ideal for students and developers studying eye-tracking and React integration.

---

## Quickstart (Beta, GitHub Version)

```bash
git clone https://github.com/neelmajmudar/FullFocus.git
cd FullFocus
npm install
npm start
