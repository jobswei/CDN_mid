import cv2
import os
import tqdm 

image_folder = '/home/airport/CDN_mod/outputs_video05/'
video_name = '102016_demo.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = sorted(images, key=lambda k: int(k[:-4].split('_')[-1]))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 4, (width, height))
for image in tqdm.tqdm(images):
    video.write(cv2.imread(os.path.join(image_folder, image)))
video.release()
