import functools
import os
import re
import subprocess
import shutil

import cv2
import numpy as np
from PIL import Image

"""
sudo add-apt-repository ppa:djcj/hybrid
sudo apt-get update
sudo apt-get install ffmpeg
"""
'''
local_ffmpeg_path = '/home/daivd/PycharmProjects/ffmpeg-3.3.4/bin-linux/ffmpeg'
gpu_ffmpeg_path = '/storage/dldi/PyProjects/ffmpeg-3.3.4/bin-linux/ffmpeg'
'''

local_ffmpeg_path = 'ffmpeg'
gpu_ffmpeg_path = 'ffmpeg'


def extract_frames(video_file, num_frames=8):
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        pass
    if num_frames == -1:
        return extract_all_frames(video_file)

    output = subprocess.Popen(['ffmpeg', '-i', video_file],
                              stderr=subprocess.PIPE).communicate()
    # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    re_duration = re.compile('Duration: (.*?)\.')
    duration = re_duration.search(str(output[1])).groups()[0]

    seconds = functools.reduce(lambda x, y: x * 60 + y,
                               map(int, duration.split(':')))
    rate = num_frames / float(seconds)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-vf', 'fps={}'.format(rate),
                               '-vframes', str(num_frames),
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    # print(output)
    extract_frame_paths = sorted([os.path.join('frames', frame)
                                  for frame in os.listdir('frames')])

    res_frames = load_frames(extract_frame_paths)
    # subprocess.call(['rm', '-rf', 'frames'])
    return res_frames, extract_frame_paths


def extract_all_frames(video_file, image_dir):
    extract_frame_flag = False
    if os.path.exists(image_dir):
        # check whether the frames right
        video_cap = cv2.VideoCapture(video_file)

        frame_count = 0
        while True:
            ret, frame = video_cap.read()
            if ret is False:
                break
            frame_count += 1

        open_cv_frames = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
        if abs(open_cv_frames - frame_count) > 1:
            print(open_cv_frames, frame_count, 'Deleting: ', image_dir)
            extract_frame_flag = True
            shutil.rmtree(image_dir)
        else:
            print(video_file, "frames are correct!")
    else:
        extract_frame_flag = True

    if extract_frame_flag:
        try:
            os.makedirs(image_dir)
        except OSError:
            pass
        os.system(gpu_ffmpeg_path + ' -i ' + video_file + ' ' + image_dir + '/%4d.jpg')


def load_frames(frame_paths, num_frames=8):
    frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


def parallel_extract_frames(video_root_dir, img_root_dir):
    for each_vid_root_dir in os.listdir(video_root_dir):
        video_list = os.listdir(os.path.join(video_root_dir, each_vid_root_dir))
        for each_idx, each_vid in enumerate(video_list):
            print('=' * 20, (each_idx, len(video_list)), '=' * 20)
            each_vid_frame_dir = each_vid[:-4] + '_frames'
            # print(each_vid_frame_dir)
            img_path = os.path.join(img_root_dir, each_vid_root_dir, each_vid_frame_dir)

            try:
                os.makedirs(os.path.join(img_root_dir, each_vid_root_dir))
            except OSError:
                pass

            if each_vid_frame_dir not in os.listdir(os.path.join(img_root_dir, each_vid_root_dir)):
                try:
                    print("=" * 20, 'Mkdir: ', img_path)
                    os.makedirs(img_path)
                except OSError:
                    pass

                each_vid_path = os.path.join(video_root_dir, each_vid_root_dir, each_vid)

                extract_all_frames(each_vid_path, img_path)

            else:
                print("=" * 20, img_path, 'is exist! ', "=" * 20)


def parallel_ext_v1_frames(video_path, img_root_dir):
    for each_vid in os.listdir(video_path):
        each_img_path = os.path.join(img_root_dir, each_vid[:-4])

        if each_img_path not in os.listdir(img_root_dir):
            try:
                os.makedirs(each_img_path)
            except OSError:
                pass

            extract_all_frames(os.path.join(video_path, each_vid), each_img_path)


if __name__ == '__main__':
    #gpu_path = './vidor-dataset/video_test/test'
    #gpu_path = './vidor-dataset/video_test/test'
    #local_vidor_frames_path = './vidor-dataset/vide_test/frames'
    #local_vidor_frames_path = './vidor-dataset/video_test/frames'

    #local_path = './vidor-dataset/video_train/video' : train
    local_path = './vidor-dataset/video_test/video'

    for root, dirs, files in os.walk(local_path):
        for each_idx, each_file in enumerate(files):
            print('=' * 20, (each_idx, len(files)), '=' * 20)
            each_vid_path = os.path.join(root, each_file)
            video_path_splits = each_vid_path.split('/')
            image_dir = os.path.join('data/frames', video_path_splits[-2], video_path_splits[-1][:-4] + '_frames')
            extract_all_frames(each_vid_path, image_dir)


    # parallel_extract_frames(gpu_path, local_vidor_frames_path)

    # parallel_ext_v1_frames(local_vidvrd_path, local_vidvrd_frames_path)