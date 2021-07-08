import argparse
import os
import time
from distutils.util import strtobool
import cv2
from tqdm import tqdm
from model.detector.faster_rcnn_x101 import FasterRCNNX101
from model.tracker.deep_sort.deep_sort import DeepSort
import json
from annotation import write_json


class AnnotationModel(object):
    def __init__(self, args):
        self.args = args
        self.detector = FasterRCNNX101()
        self.deep_sort = DeepSort(self.detector)

    def detect_image_list(self, output_dir, image_path, anno_path):
        """
        :param output_dir: '/media/ailab/4f12b7e1-629d-424c-8fe5-35a9c2f8db76/jonghun/workspace/vidvrd/vidvrd-dataset/track_annotation/train'
        :param image_path: '/media/ailab/4f12b7e1-629d-424c-8fe5-35a9c2f8db76/jonghun/workspace/vidvrd/vidvrd-dataset/frames/ILSVRC2015_train_00005003'
        :param anno_path: '/media/ailab/4f12b7e1-629d-424c-8fe5-35a9c2f8db76/jonghun/workspace/vidvrd/vidvrd-dataset/train/ILSVRC2015_train_00005003.json'
        :return:
        """
        # 물체 추적 모델은 이미지를 새로 추적 시에 초기화 필요
        self.deep_sort = DeepSort(self.detector)
        image_list = self.get_image_list(image_path)
        total_video_track_list = []

        print("\ntrack all images")

        for image in tqdm(image_list):
            image_np = cv2.imread(os.path.join(image_path, image))
            boxes, classes, scores = self.detector.detect(image_np)

            if boxes.shape[0] != 0:
                outputs = self.deep_sort.update(image_np, boxes, classes, scores)

                if str(type(outputs)) == "<class 'numpy.ndarray'>":
                    outputs = outputs.tolist()

            else:
                outputs = []

            total_video_track_list.append(outputs)

        print("write json")
        width, frame_count, video_id, fps, height = write_json.get_anno_info(anno_path)
        # 수정
        print(video_id, frame_count, len(total_video_track_list))
        # 추후에 ffmpeg 버전 문제인지 확인 필요
        # len(image_list) 는 video 를 ffmpeg 로 자른건데 하나씩 더 많은 경우 많음
        # frame_count 와 len(image_list) 임의적으로 동일하게 맞춤
        if frame_count < len(total_video_track_list):
            tmp = len(total_video_track_list) - frame_count
            total_video_track_list = total_video_track_list[:-tmp]
        # 해당 예외는 발생한적 없음
        if frame_count > len(total_video_track_list):
            raise Exception('frame_count > total_video_track_list')
        subject_object_dict_list = write_json.get_subject_objects(total_video_track_list)
        total_trajectories = write_json.get_trajectories(total_video_track_list)
        write_json.check_error(frame_count, total_video_track_list, subject_object_dict_list, total_trajectories)

        json_dict = {
            "width": width,
            "frame_count": frame_count,
            "video_id": video_id,
            "fps": fps,
            "trajectories": total_trajectories,
            "subject/objects": subject_object_dict_list,
            "height": height
        }

        video_folder = anno_path.split('/')[-2]
        video_name = anno_path.split('/')[-1]

        if not os.path.exists(os.path.join(output_dir, video_folder)):
            os.mkdir(os.path.join(output_dir, video_folder))

        json_output_path = os.path.join(output_dir, video_folder, video_name)

        with open(json_output_path, 'w') as fout:
            json.dump(json_dict, fout)

    def get_image_list(self, image_path):
        image_list = os.listdir(image_path)
        image_list = sorted(image_list)

        # error check
        for i in range(0, len(image_list) - 1):
            i_idx = int(image_list[i].split('.')[0])
            next_i_idx = int(image_list[i + 1].split('.')[0])
            assert i_idx + 1 == next_i_idx

        return image_list

def write_all_json(annoation_model, args, mode):
    """
    :param annoation_model:
    :param args:
    :param mode: 'train' or 'test'
    :return:
    """
    output_root = args.output_root
    image_root = args.image_root
    anno_root = args.anno_root

    output_dir = os.path.join(output_root, mode)
    image_dir = os.path.join(image_root)
    anno_dir = os.path.join(anno_root, mode)
    anno_file_list = os.listdir(anno_dir)

    print("get anno file")
    cnt = 0

    for anno_file in anno_file_list:
        cnt += 1
        print("current {} anno : {}/{}".format(mode, cnt, len(anno_file_list)))
        print(anno_file)

        anno_path = os.path.join(anno_dir, anno_file)
        video_path = os.path.join(image_dir, anno_file.split('.')[0])
        video_frame = os.listdir(video_path)
        for vf in video_frame :
            image_path = os.path.join(video_path, vf)
            anno_tmp = '{}/{}.json'.format(anno_path, vf.split('_')[0])
            annoation_model.detect_image_list(output_dir, image_path, anno_tmp)


def write_one_json(annoation_model, args, mode):
    output_root = args.output_root
    image_root = args.image_root
    anno_root = args.anno_root

    # anno_name = "0000/2401075277.json"
    anno_name = args.anno_name

    output_dir = os.path.join(output_root)
    image_dir = os.path.join(image_root)
    anno_dir = os.path.join(anno_root, mode)

    anno_path = os.path.join(anno_dir, anno_name)
    # anno_path : ./vidor-dataset/training/1111/289254713.json
    image_name = anno_name.split('.')[0]
    image_name = '{}_frames'.format(image_name)
    image_path = os.path.join(image_dir, image_name)
    # image_path : ./vidor-dataset/frames/1111/2892594713_frame
    annoation_model.detect_image_list(output_dir, image_path, anno_path)

