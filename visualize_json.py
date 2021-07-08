import argparse
import os
import json
import cv2
import numpy as np
from utils.utils import draw_bboxes
from annotation.utils import get_category_id_dict


def get_tid_category_dict(subject_objects):
    tid_category_dict = {}

    for dictionary in subject_objects:
        category = dictionary['category']
        tid = dictionary['tid']

        tid_category_dict[tid] = category

    return tid_category_dict


def write_images(args):
    folder_name = args.folder_name
    image_dir = args.image_dir
    video_path = args.video_path
    anno_dir = args.anno_dir

    anno_path = os.path.join(anno_dir, '{}'.format(folder_name))
    category_id_dict = get_category_id_dict()

    with open(anno_path) as anno_name:
        json_data = json.load(anno_name)
        video_id = json_data['video_id']
        total_trajectories = json_data['trajectories']
        subject_objects = json_data['subject/objects']
        fps = int(json_data['fps'])
        width = json_data['width']
        height = json_data['height']

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        tid_category_dictionary = get_tid_category_dict(subject_objects)

        for i in range(len(total_trajectories)):
            print("{}/{}".format(i, len(total_trajectories)))
            trajectories = total_trajectories[i]
            image_name = '{}.jpg'.format(str(i + 1).zfill(4))
            fold_num = folder_name.split('/')[0]
            file_num = folder_name.split('/')[1].split('.')[0]
            file_num = '{}_frames'.format(file_num)

            image_path = os.path.join(image_dir, fold_num, file_num, image_name)

            print(image_path)

            image_numpy = cv2.imread(image_path)

            output_list = []

            for trajectory in trajectories:
                xmin = trajectory["bbox"]["xmin"]
                ymin = trajectory["bbox"]["ymin"]
                xmax = trajectory["bbox"]["xmax"]
                ymax = trajectory["bbox"]["ymax"]
                tid = trajectory["tid"]
                class_name = tid_category_dictionary[tid]
                class_id = category_id_dict[class_name]

                output_list.append([xmin, ymin, xmax, ymax, tid, class_id])

            if len(output_list) > 0:
                outputs = np.array(output_list)
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                class_id = outputs[:, -1]
                image_numpy = draw_bboxes(image_numpy, bbox_xyxy, identities, class_id)

            video_writer.write(image_numpy)


if __name__ == '__main__':
    folder_name = "1111/2892594713.json"
    #image_dir = "/media/ailab/4f12b7e1-629d-424c-8fe5-35a9c2f8db76/jonghun/workspace/vidvrd/vidor-dataset/frame"
    image_dir = "./vidor-dataset/frames"
    output_dir = "./outputs/visualize_json/output"
    video_name = folder_name.split('/')[1].split('.')[0]
    video_name = '{}.avi'.format(video_name)
    video_path = os.path.join(output_dir, video_name)
    anno_dir = "./outputs/output_json"
    mode = "train"

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, default=folder_name)
    parser.add_argument("--image_dir", type=str, default=image_dir)
    parser.add_argument("--video_path", type=str, default=video_path)
    parser.add_argument("--anno_dir", type=str, default=anno_dir)
    parser.add_argument("--mode", type=str, default=mode)
    args = parser.parse_args()

    print("write image")
    write_images(args)

