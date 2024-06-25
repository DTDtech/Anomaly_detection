import cv2

import subprocess

import os

input_file = 'data/v_Bowling_g01_c01.avi'
cwd = os.getcwd()

# run terminal command to convert avi video into mp4 video
# subprocess.run(['ffmpeg',
#                 '-i',
#                 input_file,
#                 '-qscale',
#                 '0',
#                 'data/v_Bowling_g01_c01.mp4',
#                 '-loglevel',
#                 'quiet'])

video_path = 'data/test.mp4'

extracted_frames_folder = cwd + "/extracted_frames"

def load_video(video_path):
    frames = []

    cap = cv2.VideoCapture(video_path)

    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))

    FRAME_SAVE_PER_SECOND = 3
    frame_save_counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            break

        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
        if frame_save_counter < FRAME_SAVE_PER_SECOND:
            file_name = extracted_frames_folder + "/frame_" + str(frame_id) + ".jpg"
            cv2.imwrite(file_name, frame)
            frame_save_counter += 1

        if (frame_id % frame_rate == 0): 
            frame_save_counter = 0

        frames.append(frame)

    cap.release()
    return frames

load_video(video_path)





