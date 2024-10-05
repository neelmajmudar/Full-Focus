# Full-Focus
Eye Gaze tracker to measure how distracted/focused the user is.

What inspired this project and how I brought it to life: 

This was a really fun project I did during the Summer of 2023. While I would do my work, I kept noticing that I would easily get distracted as I'd constantly reach for my phone. So, to counteract this, I made an OpenCV project that would help me stay focused. I looked through Google's documentation on Mediapipe and several YouTube tutorials online to figure out how to draw a mesh grid of my face on OpenCV. After figuring out the coordinates of the points drawn on my face, I reduced them to just my eyes and used a simple distance formula to keep track of the distance from the center of my pupil to each end of my eye. With this, I was able to set a threshold that would alert the user when they are looking anywhere outside the range of their computer screen from up, down, left, right, and closed-eye movements. I used tkinter to create a simple, easy-to-use GUI for this program and allowed users to change various thresholds so it could be more suited to their setup/needs. As an addition, each time the user was distracted, I even added a function to play audio in the background to remind the user to get back to work. Once the user quits the application, they can even see various metrics like the total time elapsed, time distracted, percent time focused, etc.

Demo video is too big to post here so please take a look at it here: https://imgur.com/T4zbWTq
