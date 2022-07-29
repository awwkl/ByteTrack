import os
import numpy as np
import cv2

video_list = []
video_list.append('MOT17-04')       # Elevated viewpoint, Highest density
video_list.append('MOT17-10')       # Night, Person moving camera
video_list.append('MOT17-13')       # Bus moving camera + Lots of motion and turning

fps_dict = {'MOT17-13':	25, 
            'MOT17-05': 14}

def create_video(input_img_dir, output_path, fps, max_frames=None):
    filenames = os.listdir(input_img_dir)
    if max_frames:
        filenames = filenames[:max_frames]

    img_array = []
    for fname in filenames:
        img_path = os.path.join(input_img_dir, fname)
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


train_videos_dir = 'C:/Users/Khai_Loong/Documents/mmtracking/data/MOT17/train/'
train_videos = list( os.listdir(train_videos_dir) )
train_videos = [vid for vid in train_videos if vid.endswith('FRCNN')]

# ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
for vid in train_videos:
    input_img_dir = os.path.join(train_videos_dir, vid, 'img1')
    vid = vid.replace('-FRCNN', '')
    output_path = os.path.join('C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/input_videos/frames150', vid + '.mp4')

    fps = fps_dict[vid] if vid in fps_dict else 30

    print(f'{vid}: {fps}')
    max_frames = 150
    create_video(input_img_dir, output_path, fps, max_frames)
