# from cProfile import label
# import cv2
# import mediapipe as mp
# import pandas as pd
# import numpy
# import keras
# import threading
# import os

# cap = cv2.VideoCapture(0)

# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

# model = keras.models.load_model("lstm-model.h5")

# lm_list = []

# # Create the "screenshots" folder if it doesn't exist
# if not os.path.exists("screenshots"):
#     os.makedirs("screenshots")

# def make_landmark_timestep(results):
#     print(results.pose_landmarks.landmark)
#     c_lm = []
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         c_lm.append(lm.x)
#         c_lm.append(lm.y)
#         c_lm.append(lm.z)
#         c_lm.append(lm.visibility)
#     return c_lm

# def draw_landmark_on_image(mpDraw, results, frame):
#     mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         h, w, c = frame.shape
#         print(id, lm)
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
#     return frame

# def draw_class_on_image(label, img):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     bottomLeftCornerOfText = (10,30)
#     fontScale = 1
#     if label == "punch":
#         fontColor = (0,0,255)
#     else:
#         fontColor = (0,255,0)
#     thickness = 2
#     lineType = 2
#     cv2.putText(img, str(label),
#                 bottomLeftCornerOfText,
#                 font,
#                 fontScale,
#                 fontColor,
#                 thickness,
#                 lineType)
#     return img

# def detect(model, lm_list):
#     global label
#     lm_list = numpy.array(lm_list)
#     lm_list = numpy.expand_dims(lm_list, axis=0)
#     result = model.predict(lm_list)
#     if result[0][0] > 0.5:
#         label = "punch"
#     else:
#         label = "neutral"
#     return str(label)

# i = 0
# warm_up_frames = 60
# screenshot_interval = 60

# while True:
#     ret, frame = cap.read()
#     frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frameRGB)
#     i=i+1
#     if i > warm_up_frames:
#         print("Start detecting...")
#         if results.pose_landmarks:
#             lm = make_landmark_timestep(results)
#             lm_list.append(lm)
#             if len(lm_list) == 20:
#                 t1 = threading.Thread(target=detect, args=(model, lm_list, ))
#                 t1.start()
#                 t1.join()  # Wait for the thread to finish
#                 lm_list = []
#             x_coordinate = list()
#             y_coordinate = list()
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 x_coordinate.append(cx)
#                 y_coordinate.append(cy)
#             if label == "neutral":
#                 cv2.rectangle(img=frame,
#                                 pt1=(min(x_coordinate), max(y_coordinate)),
#                                 pt2=(max(x_coordinate), min(y_coordinate)-25),
#                                 color=(0,255,0),
#                                 thickness=1)
#             elif label == "punch":
#                 cv2.rectangle(img=frame,
#                                 pt1=(min(x_coordinate), max(y_coordinate)),
#                                 pt2=(max(x_coordinate), min(y_coordinate)-25),
#                                 color=(0,0,255),
#                                 thickness=3)

#             frame = draw_landmark_on_image(mpDraw, results, frame)
#         frame = draw_class_on_image(label, frame)
#         cv2.imshow("image", frame)
        
#          # Take and save a screenshot at regular intervals
#         if i % screenshot_interval == 0:
#             screenshot_filename = os.path.join("screenshots", f"screenshot_{int(i / screenshot_interval)}.png")
#             cv2.imwrite(screenshot_filename, frame)
#             print(f"Screenshot saved: {screenshot_filename}")
            
#         if cv2.waitKey(1) == ord('q'):
#             break

# df = pd.DataFrame(lm_list)
# df.to_csv(label+".txt")
# cap.release()
# cv2.destroyAllWindows()



###################################################################################################
                        #THIS IS MAIN CODE#
                        
# from cProfile import label  
# import cv2
# import mediapipe as mp
# import pandas as pd
# import numpy
# import keras
# import threading
# import os
# import time

# cap = cv2.VideoCapture(0)

# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

# model = keras.models.load_model("lstm-model.h5")

# lm_list = []

# # Create the "screenshots" and "punch_screenshots" folders if they don't exist
# if not os.path.exists("screenshots"):
#     os.makedirs("screenshots")
# if not os.path.exists("punch_screenshots"):
#     os.makedirs("punch_screenshots")

# def make_landmark_timestep(results):
#     print(results.pose_landmarks.landmark)
#     c_lm = []
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         c_lm.append(lm.x)
#         c_lm.append(lm.y)
#         c_lm.append(lm.z)
#         c_lm.append(lm.visibility)
#     return c_lm

# def draw_landmark_on_image(mpDraw, results, frame):
#     mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         h, w, c = frame.shape
#         print(id, lm)
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
#     return frame

# def draw_class_on_image(label, img):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     bottomLeftCornerOfText = (10,30)
#     fontScale = 1
#     if label == "punch":
#         fontColor = (0,0,255)
#     else:
#         fontColor = (0,255,0)
#     thickness = 2
#     lineType = 2
#     cv2.putText(img, str(label),
#                 bottomLeftCornerOfText,
#                 font,
#                 fontScale,
#                 fontColor,
#                 thickness,
#                 lineType)
#     return img

# def detect(model, lm_list):
#     global label
#     lm_list = numpy.array(lm_list)
#     lm_list = numpy.expand_dims(lm_list, axis=0)
#     result = model.predict(lm_list)
#     if result[0][0] > 0.5:
#         label = "punch"
#         return True
#     else:
#         label = "neutral"
#         return False

# i = 0
# warm_up_frames = 60
# screenshot_interval = 80  # take a screenshot every 60 frames (adjust as needed)

# while True:
#     ret, frame = cap.read()
#     frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frameRGB)
#     i = i + 1
#     if i > warm_up_frames:
#         print("Start detecting...")
#         if results.pose_landmarks:
#             lm = make_landmark_timestep(results)
#             lm_list.append(lm)
#             if len(lm_list) == 20:
#                 t1 = threading.Thread(target=detect, args=(model, lm_list, ))
#                 t1.start()
#                 t1.join()  # Wait for the thread to finish
#                 lm_list = []

#             x_coordinate = list()
#             y_coordinate = list()
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 x_coordinate.append(cx)
#                 y_coordinate.append(cy)

#             if label == "punch":
#                 punch_screenshot_filename = os.path.join("punch_screenshots", f"screenshot_{int(time.time())}.png")
#                 cv2.imwrite(punch_screenshot_filename, frame)
#                 print(f"Punch Screenshot saved: {punch_screenshot_filename}")

#                 cv2.rectangle(img=frame,
#                               pt1=(min(x_coordinate), max(y_coordinate)),
#                               pt2=(max(x_coordinate), min(y_coordinate)-25),
#                               color=(0, 0, 255),
#                               thickness=3)

#             elif label == "neutral":
#                 cv2.rectangle(img=frame,
#                               pt1=(min(x_coordinate), max(y_coordinate)),
#                               pt2=(max(x_coordinate), min(y_coordinate)-25),
#                               color=(0, 255, 0),
#                               thickness=1)

#             frame = draw_landmark_on_image(mpDraw, results, frame)

#         frame = draw_class_on_image(label, frame)
#         cv2.imshow("image", frame)

#         # Take and save a screenshot when punch is detected
#         if label == "punch":
#             punch_screenshot_filename = os.path.join("punch_screenshots", f"screenshot_{int(time.time())}.png")
#             cv2.imwrite(punch_screenshot_filename, frame)
#             print(f"Punch Screenshot saved: {punch_screenshot_filename}")

#         if cv2.waitKey(1) == ord('q'):
#             break

# df = pd.DataFrame(lm_list)
# df.to_csv(label + ".txt")
# cap.release()
# cv2.destroyAllWindows()


######################################################################################################################################
                                                    # 2nd main code with deblurring




# from cProfile import label  
# import cv2
# import mediapipe as mp
# import pandas as pd
# import numpy as np
# import keras
# import threading
# import os
# import time
# from PIL import Image, ImageEnhance, ImageFilter

# # Load the pre-trained model
# model = keras.models.load_model("lstm-model.h5")

# # Video capture setup
# cap = cv2.VideoCapture(0)

# # MediaPipe setup
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

# lm_list = []

# # Create folders if they don't exist
# if not os.path.exists("screenshots"):
#     os.makedirs("screenshots")
# if not os.path.exists("punch_screenshots"):
#     os.makedirs("punch_screenshots")

# # Function to make a timestep of landmark data
# def make_landmark_timestep(results):
#     c_lm = []
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         c_lm.append(lm.x)
#         c_lm.append(lm.y)
#         c_lm.append(lm.z)
#         c_lm.append(lm.visibility)
#     return c_lm

# # Function to draw landmarks on the image
# def draw_landmark_on_image(mpDraw, results, frame):
#     mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         h, w, c = frame.shape
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
#     return frame

# # Function to draw classification label on the image
# def draw_class_on_image(label, img):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     bottomLeftCornerOfText = (10, 30)
#     fontScale = 1
#     if label == "punch":
#         fontColor = (0, 0, 255)  # Red color for punch
#     else:
#         fontColor = (0, 255, 0)  # Green color for neutral
#     thickness = 2
#     lineType = 2
#     cv2.putText(img, str(label),
#                 bottomLeftCornerOfText,
#                 font,
#                 fontScale,
#                 fontColor,
#                 thickness,
#                 lineType)
#     return img

# # Function to detect punch
# def detect(model, lm_list):
#     global label
#     lm_list = np.array(lm_list)
#     lm_list = np.expand_dims(lm_list, axis=0)
#     result = model.predict(lm_list)
#     if result[0][0] > 0.5:
#         label = "punch"
#         return True
#     else:
#         label = "neutral"
#         return False

# # Image enhancement function
# def enhance_image(image):
#     # 1. Detail Enhancement (Unsharp Masking)
#     unsharp_mask_radius = 2
#     unsharp_mask_amount = 150
#     detail_enhanced_image = image.filter(
#         ImageFilter.UnsharpMask(radius=unsharp_mask_radius, percent=unsharp_mask_amount)
#     )

#     # 2. Selective Sharpening
#     sharpness_factor = 3.0
#     enhancer_sharpness = ImageEnhance.Sharpness(detail_enhanced_image)
#     sharpened_image = enhancer_sharpness.enhance(sharpness_factor)

#     # 3. Color Adjustment (Saturation and Contrast)
#     saturation_factor = 1.1
#     enhancer_saturation = ImageEnhance.Color(sharpened_image)
#     saturated_image = enhancer_saturation.enhance(saturation_factor)

#     contrast_factor = 1.1
#     enhancer_contrast = ImageEnhance.Contrast(saturated_image)
#     final_image = enhancer_contrast.enhance(contrast_factor)

#     return final_image

# # Main loop
# i = 0
# warm_up_frames = 60
# while True:
#     ret, frame = cap.read()
#     frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frameRGB)
#     i += 1
#     if i > warm_up_frames:
#         print("Start detecting...")
#         if results.pose_landmarks:
#             lm = make_landmark_timestep(results)
#             lm_list.append(lm)
#             if len(lm_list) == 20:
#                 t1 = threading.Thread(target=detect, args=(model, lm_list,))
#                 t1.start()
#                 t1.join()  # Wait for the thread to finish
#                 lm_list = []

#             if label == "punch":
#                 punch_screenshot_filename = os.path.join("punch_screenshots", f"screenshot_{int(time.time())}.png")
#                 pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 enhanced_frame = enhance_image(pil_frame)
#                 enhanced_frame.save(punch_screenshot_filename)
#                 print(f"Punch Screenshot saved: {punch_screenshot_filename}")

#             elif label == "neutral":
#                 print("Neutral detected. No action taken.")
                
#             frame = draw_landmark_on_image(mpDraw, results, frame)

#         frame = draw_class_on_image(label, frame)
#         cv2.imshow("image", frame)

#         if cv2.waitKey(1) == ord('q'):
#             break

# # Save landmark data to a file
# df = pd.DataFrame(lm_list)
# df.to_csv(label + ".txt")
# cap.release()
# cv2.destroyAllWindows()

################################################################################################################### 
#                                         3rd main code with punch detections ss

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


# from cProfile import label
# from tkinter.ttk import Frame  
# from SENDING import send_email
# import cv2
# import mediapipe as mp
# import pandas as pd
# import numpy as np
# import keras
# import threading
# import os
# import time
# import asyncio
# from PIL import Image, ImageEnhance, ImageFilter

# # Load the pre-trained model
# model = keras.models.load_model("lstm-model.h5")

# # Video capture setup
# cap = cv2.VideoCapture(0)

# # MediaPipe setup
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

# lm_list = []

# # Create folders if they don't exist
# if not os.path.exists("screenshots"):
#     os.makedirs("screenshots")
# if not os.path.exists("punch_screenshots"):
#     os.makedirs("punch_screenshots")

# # Function to make a timestep of landmark data
# def make_landmark_timestep(results):
#     c_lm = []
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         c_lm.append(lm.x)
#         c_lm.append(lm.y)
#         c_lm.append(lm.z)
#         c_lm.append(lm.visibility)
#     return c_lm

# # Function to draw landmarks on the image
# def draw_landmark_on_image(mpDraw, results, frame):
#     mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         h, w, c = frame.shape
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
#     return frame

# # Function to draw classification label on the image
# def draw_class_on_image(label, img):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     bottomLeftCornerOfText = (10, 30)
#     fontScale = 1
#     if label == "punch":
#         fontColor = (0, 0, 255)  # Red color for punch
#     else:
#         fontColor = (0, 255, 0)  # Green color for neutral
#     thickness = 2
#     lineType = 2
#     cv2.putText(img, str(label),
#                 bottomLeftCornerOfText,
#                 font,
#                 fontScale,
#                 fontColor,
#                 thickness,
#                 lineType)
#     return img

# # Function to detect punch
# def detect(model, lm_list):
#     global label
#     lm_list = np.array(lm_list)
#     lm_list = np.expand_dims(lm_list, axis=0)
#     result = model.predict(lm_list)
#     if result[0][0] > 0.5:
#         label = "punch"
#         return True
#     else:
#         label = "neutral"
#         return False

# # Image enhancement function
# def enhance_image(image):
#     # 1. Detail Enhancement (Unsharp Masking)
#     unsharp_mask_radius = 2
#     unsharp_mask_amount = 150
#     detail_enhanced_image = image.filter(
#         ImageFilter.UnsharpMask(radius=unsharp_mask_radius, percent=unsharp_mask_amount)
#     )

#     # 2. Selective Sharpening
#     sharpness_factor = 3.0
#     enhancer_sharpness = ImageEnhance.Sharpness(detail_enhanced_image)
#     sharpened_image = enhancer_sharpness.enhance(sharpness_factor)

#     # 3. Color Adjustment (Saturation and Contrast)
#     saturation_factor = 1.1
#     enhancer_saturation = ImageEnhance.Color(sharpened_image)
#     saturated_image = enhancer_saturation.enhance(saturation_factor)

#     contrast_factor = 1.1
#     enhancer_contrast = ImageEnhance.Contrast(saturated_image)
#     final_image = enhancer_contrast.enhance(contrast_factor)

#     return final_image

# # Function to send email with punch screenshots
# def send_email_with_punch_screenshot(frame):
#     global label
#     if label == "punch":
#             punch_screenshot_filename = os.path.join("punch_screenshots", f"screenshot_{int(time.time())}.png")
#             pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             enhanced_frame = enhance_image(pil_frame)
#             enhanced_frame.save(punch_screenshot_filename)
#             print(f"Punch Screenshot saved: {punch_screenshot_filename}")

#             # Send email with the screenshot
#             receiver_email = "pooja.mestry67@gmail.com"
#             subject = "**VIOLENCE ALERT**"
#             body = "Please find attached screenshots of the incident."
#             attachment_folder = "punch_screenshots"
#             send_email(receiver_email, subject, body, attachment_folder)

# # Main loop
# def main_loop():
#     global lm_list
#     # global label
#     i = 0
#     warm_up_frames = 60
#     while True:
#         ret, frame = cap.read()
#         frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frameRGB)
#         i += 1
#         if i > warm_up_frames:
#             print("Start detecting...")
#             if results.pose_landmarks:
#                 lm = make_landmark_timestep(results)
#                 lm_list.append(lm)
#                 if len(lm_list) == 20:
#                     is_punch = detect(model, lm_list)
#                     lm_list = []
                    
#                     if is_punch:
#                         label = "punch"
#                     else:
#                         label = "neutral"

#                 # if is_punch:
#                 #     punch_screenshot_filename = os.path.join("punch_screenshots", f"screenshot_{int(time.time())}.png")
#                 #     pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 #     enhanced_frame = enhance_image(pil_frame)
#                 #     enhanced_frame.save(punch_screenshot_filename)
#                 #     print(f"Punch Screenshot saved: {punch_screenshot_filename}")
                
#                 #  # Send email with the screenshot
#                 #     receiver_email = "pooja.mestry67@gmail.com"
#                 #     subject = "**VIOLENCE ALERT**"
#                 #     body = "Please find attached screenshots of the incident."
#                 #     attachment_folder = "punch_screenshots"
#                 #     send_email(receiver_email, subject, body, attachment_folder)
                    
#                 # if label == "neutral":
#                 #     print("Neutral detected. No action taken.")
#                     send_email_with_punch_screenshot(frame)
                    
#                 frame = draw_landmark_on_image(mpDraw, results, frame)

#             frame = draw_class_on_image(label, frame)
#             cv2.imshow("image", frame)

#             if cv2.waitKey(1) == ord('q'):
#                 break
            
# # async def main():
# #     # Start the punch detection loop
# #     await punch_detection()

# # # Run the event loop
# # asyncio.run(main())
            
# # Start punch detection in a separate thread
# # punch_detection_thread = threading.Thread(target=punch_detection)
# # punch_detection_thread.start()

# # Start email sending in a separate thread
# # send_email_thread = threading.Thread(target=send_email_with_punch_screenshot)
# # send_email_thread.start()

# # Wait for the threads to finish
# # punch_detection_thread.join()
# # send_email_thread.join()

# # Wait for the punch detection thread to finish
# # punch_detection_thread.join()

# # Start the main loop in a separate thread
# main_thread = threading.Thread(target=main_loop)
# main_thread.start()

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# # # Save landmark data to a file
# # df = pd.DataFrame(lm_list)
# # df.to_csv(label + ".txt")
# # cap.release()
# # cv2.destroyAllWindows()




















       







