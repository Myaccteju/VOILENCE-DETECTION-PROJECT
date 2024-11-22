################################################################################################################### 
#                                         Main code with punch detections ss

from cProfile import label  
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import keras
import threading
import os
import time
from PIL import Image, ImageEnhance, ImageFilter

model = keras.models.load_model("lstm-model.h5")

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []

if not os.path.exists("screenshots"):
    os.makedirs("screenshots")
if not os.path.exists("punch_screenshots"):
    os.makedirs("punch_screenshots")


def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    if label == "punch":
        fontColor = (0, 0, 255)  # Red color for punch
    else:
        fontColor = (0, 255, 0)  # Green color for neutral
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

# Function to detect punch
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    if result[0][0] > 0.5:
        label = "punch"
        return True
    else:
        label = "neutral"
        return False

# Image enhancement function
def enhance_image(image):
    
    unsharp_mask_radius = 2
    unsharp_mask_amount = 150
    detail_enhanced_image = image.filter(
        ImageFilter.UnsharpMask(radius=unsharp_mask_radius, percent=unsharp_mask_amount)
    )

    
    sharpness_factor = 3.0
    enhancer_sharpness = ImageEnhance.Sharpness(detail_enhanced_image)
    sharpened_image = enhancer_sharpness.enhance(sharpness_factor)

    
    saturation_factor = 1.1
    enhancer_saturation = ImageEnhance.Color(sharpened_image)
    saturated_image = enhancer_saturation.enhance(saturation_factor)

    contrast_factor = 1.1
    enhancer_contrast = ImageEnhance.Contrast(saturated_image)
    final_image = enhancer_contrast.enhance(contrast_factor)

    return final_image

i = 0
warm_up_frames = 60
while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    i += 1
    if i > warm_up_frames:
        print("Start detecting...")
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                is_punch = detect(model, lm_list)
                lm_list = []

                if is_punch:
                    punch_screenshot_filename = os.path.join("punch_screenshots", f"screenshot_{int(time.time())}.png")
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    enhanced_frame = enhance_image(pil_frame)
                    enhanced_frame.save(punch_screenshot_filename)
                    print(f"Punch Screenshot saved: {punch_screenshot_filename}")

            if label == "neutral":
                print("Neutral detected. No action taken.")
                
            frame = draw_landmark_on_image(mpDraw, results, frame)

        frame = draw_class_on_image(label, frame)
        cv2.imshow("image", frame)

        if cv2.waitKey(1) == ord('q'):
            break


df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()

######################################################################################################################################################




















       







