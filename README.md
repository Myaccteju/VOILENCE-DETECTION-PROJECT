## Violence Detection and Facial Recognition System for Crime Detection ##

This project aims to enhance law enforcement capabilities by integrating violence detection and facial recognition technologies. The system detects violence in real-time, captures screenshots when punches are detected, and sends these images via email to authorities. Additionally, the system uses facial recognition to identify criminals from the images, building a database for future references. This can help law enforcement agencies track suspects and identify patterns in organized crimes by analyzing video footage.

# Key Features #
1. Violence Detection:
-Detects violent events like punches in real-time using video analytics.
-Takes a screenshot when a violent action (e.g., punch) is detected.
-Sends the screenshot to specified authorities via SMTP email.

2. Facial Recognition for Crime Detection:
-Captures faces from the screenshots and uses facial recognition to compare them against a pre-existing database.
-Matches faces with known criminals or suspects and logs their appearance times.
-Alerts authorities when a match is found, indicating the potential presence of a criminal.

3. Database Creation and Matching:
-The screenshots captured during violence events are used to create a database of faces.
-When the system encounters a match with a known criminal, it triggers a notification with the 
 associated time of occurrence.
-Continuous updates and cross-checking across multiple video sources to identify organized 
 crime patterns.

4. Video Analysis:
-Analyzes 30-second to 1-minute video clips to capture faces of people present.
-Cross-references the captured faces in subsequent videos to detect recurring appearances, 
 aiding in the identification of suspects over time.

5. Pattern Recognition:
-The system tracks patterns across different videos and identifies any recurring individuals.
-This can assist in identifying criminal activities, particularly in cases of organized crime.

# Technologies Used #
1. OpenCV: Used for detecting faces in videos and images.
2. Face Recognition Libraries: Libraries like face_recognition and dlib are used for face recognition and matching.
3. SMTP: Used for sending emails with captured screenshots to authorities.
4. Python: Primary programming language for video processing and automation
5. Machine Learning: For recognizing patterns and matching faces with known criminals.

# How it Works #
1. Violence Detection:
-The system processes video footage in real-time to detect violent actions, such as punches.
-Once a punch is detected, the system takes a screenshot and sends it to the predefined 
 authorities via email using SMTP.

2. Facial Recognition:
-The system uses facial recognition software to capture and store faces from the screenshots.
-Each captured face is stored in the database with timestamps.
-When a new image or video is processed, it is compared with the database to check if any known 
 criminal faces are detected.

3. Cross-Referencing Videos:
-Videos are analyzed for faces, and matches across multiple sources (videos) are noted.
-The system records timestamps of when criminals are identified in different videos, allowing 
 authorities to track the presence of suspects.

4. Database Matching:
-Each time a new face is detected, the system checks if it already exists in the database.
-If there is a match, the system logs the match and sends alerts to the authorities, indicating 
 the identity and timing of the appearance. in Mind:
