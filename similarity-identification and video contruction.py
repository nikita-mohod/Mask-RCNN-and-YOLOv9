# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:54:39 2024

@author: ASUS
"""

#Similarity index code

import time
import os
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim

start1 = time.time()
# Function to calculate SSIM between two images
def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2 )  # Use multichannel=True for color images

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    # You can apply additional preprocessing here if needed
    return img

# Input folder containing the frames
#input_folder = 'C:/Users/cse/FRAMESNEW_yolo5'
input_folder = 'C:/Users/ASUS/FRAMESNEW_yolo5'
parent_folder = 'parent_frames'
child_folder = 'child_frames'
threshold = 0.96  # Set your desired threshold here
frame_rate = 24  # Frames per second for the output video

# Initialize a dictionary to store parent-child relationships and their counts
parent_to_children = {}
parent_counts = {}

# List all files in the input folder
frame_files = os.listdir(input_folder)
frame_files.sort(key=lambda x: int(x[5:-4]))  # Sort the frame files numerically

num_frames = len(frame_files)
i = 0  # Index for the current frame

while i < num_frames:
    frame1_path = os.path.join(input_folder, frame_files[i])
    frame1 = load_and_preprocess_image(frame1_path)

    parent_frame_name = frame_files[i]

    children = []  # List to store child frames

    for j in range(i + 1, num_frames):
        frame2_path = os.path.join(input_folder, frame_files[j])
        frame2 = load_and_preprocess_image(frame2_path)

        similarity = calculate_ssim(frame1, frame2)

        if similarity >= threshold:
            children.append(frame_files[j])
        else:
            break  # No need to continue comparing with other frames

    # Save the parent frame to the parent folder
    parent_frame_path = os.path.join(parent_folder, parent_frame_name)
    cv2.imwrite(parent_frame_path, frame1)

    # Save the children frames to the child folder
    for child_name in children:
        child_frame_path = os.path.join(child_folder, child_name)
        cv2.imwrite(child_frame_path, load_and_preprocess_image(os.path.join(input_folder, child_name)))

    # Update the parent-child relationship
    parent_to_children[parent_frame_name] = children

    # Update the parent counts
    parent_counts[parent_frame_name] = len(children) + 1  # Include the parent frame itself

    # Move to the next frame after children frames
    i += len(children) + 1

# Create a DataFrame with the parent frame counts
df = pd.DataFrame.from_dict(parent_counts, orient='index', columns=['Child Frame Count'])

# Save the DataFrame to an Excel file
excel_file_path = 'parent_frame_counts.xlsx'
df.to_excel(excel_file_path)

print(f"Parent frame counts saved in {excel_file_path}")

# Construct a video from parent frames
output_video_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame1.shape[1], frame1.shape[0]))

for parent_frame_name in parent_to_children.keys():
    parent_frame_path = os.path.join(parent_folder, parent_frame_name)
    parent_frame = cv2.imread(parent_frame_path)
    video_writer.write(parent_frame)

video_writer.release()

print(f"Video constructed and saved as {output_video_path}")
end1 = time.time()
print('Time required for similarity index all frames - %i' %(end1-start1))





# import cv2

# # Input video file and output video file
# input_video_file = 'C:/Users/cse/123.avi'
# output_video_file = 'C:/Users/cse/output_video1.avi'

# # Open the input video file
# cap = cv2.VideoCapture(input_video_file)

# # Get the original video's frame width, height, and frames per second (fps)
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fps = int(cap.get(5))

# # Define the codec and create a VideoWriter object to save the output video
# fourcc = cv2.VideoWriter_fourcc(*'H264')  # You can use other codecs like 'XVID', 'H264', etc.
# out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

# # Iterate through the frames of the input video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize the frame (adjust the width and height as needed)
#     new_width = 640
#     new_height = 360
#     resized_frame = cv2.resize(frame, (new_width, new_height))

#     # Write the resized frame to the output video
#     out.write(resized_frame)

# # Release the video capture and writer objects
# cap.release()
# out.release()

# # Close any OpenCV windows
# cv2.destroyAllWindows()

import cv2

# Input video file and output video file
input_video_file = 'C:/Users/ASUS/output_video.avi'
output_video_file = 'C:/Users/ASUS/output_video2.avi'

# Open the input video file
cap = cv2.VideoCapture(input_video_file)

# Get the original video's frame width, height, and frames per second (fps)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define the codec for the output video (H.264 is commonly used for good quality and compression)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# Create a VideoWriter object to save the output video with the same resolution and fps
out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Write the frame to the output video
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()

# Close any OpenCV windows
cv2.destroyAllWindows()

