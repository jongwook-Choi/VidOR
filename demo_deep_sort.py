import argparse
import os
import time
from distutils.util import strtobool
import cv2
import torch

from model.detector.faster_rcnn_x101 import FasterRCNNX101
from model.tracker.deep_sort.deep_sort import DeepSort
from utils.utils import draw_bboxes

class DemoDeepSort(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        self.vdo = cv2.VideoCapture()
        self.detector = FasterRCNNX101()
        self.deep_sort = DeepSort(self.detector)

    def __enter__(self):
        assert os.path.isfile(self.args.video_path), "Error: path error"
        self.vdo.open(self.args.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        while self.vdo.grab():
            start = time.time()
            _, im = self.vdo.retrieve()
            boxes, classes, scores = self.detector.detect(im)

            if boxes.shape[0] != 0:
                outputs = self.deep_sort.update(im, boxes, classes, scores)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    class_id = outputs[:, -1]
                    im = draw_bboxes(im, bbox_xyxy, identities, class_id)

            end = time.time()
            print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

            if self.args.save_path:
                self.output.write(im)


def parse_args(test_video_path, test_save_path):
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, default=test_video_path)
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--save_path", type=str, default=test_save_path)
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    # test_video_path = './vidor-dataset/video_train/video/0000/2401075277.mp4'
    #test_video_path = './vidor-dataset/video_test'
    PATH = os.path.dirname(os.path.abspath(__file__))
    dir_name = 'vidor-dataset/video_test'
    dir_path = os.path.join(PATH, dir_name)
    for root, dirs, files in os.walk(dir_path):
        print(root)
        for file in files:
            file_path = os.path.join(root,file)
            print(file_path)
            output_video_name = '{}.avi'.format(file_path.split('/')[-1].split('.')[0])
            output_video_path = os.path.join('outputs/demo', output_video_name)

            args = parse_args(file_path, output_video_path)
            with DemoDeepSort(args) as det:
                det.detect()

