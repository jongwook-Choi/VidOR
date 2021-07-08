import argparse
import os
import cv2
from tqdm import tqdm
from model.detector.faster_rcnn_x101 import FasterRCNNX101

"""
:param output_dir: './outputs/output_txt'
:param image_path: './vidor-dataset/frames_validation'
:return:
"""

def write_txt_one(image_root, anno_name, output_root):
    detector = FasterRCNNX101()
    anno_f, anno_n = anno_name.split('/')
    image_path = f'{image_root}/{anno_f}/{anno_n}_frames'

    image_list = get_image_list(image_path)
    print("\ntrack all images")

    file = open(os.path.join(output_root,'tmp.txt'), 'w')
    for image in tqdm(image_list):
        image_np = cv2.imread(os.path.join(image_path, image))
        num = image.split('.')[0]
        # class 정보는 안들어가서 뺌
        boxes, cls, scores = detector.detect(image_np)
        for box, score in zip(boxes, scores):
            '''
            bb_left = x0 = box[0]
            bb_top = y1 = bbox[3]
            bb_width = x1-x0 = bbox[2]-bbox[0]
            bb_height = y1-y0 = bbox[3]-bbox[1]
            '''
            data = f'{int(num)},-1,{box[0]},{box[3]},{box[2]-box[0]},{box[3]-box[1]},{score},{cls}\n'
            file.write(data)
    file.close()

def write_txt_all(image_root, output_root):
    detector = FasterRCNNX101()
    for f_num in os.listdir(image_root):
        anno_f = f_num
        for video_frame in os.listdir(os.path.join(image_root,anno_f)):
            anno_n = video_frame.split('_')[0]
            image_path = f'{image_root}/{anno_f}/{anno_n}_frames'
            image_list = get_image_list(image_path)
            print(f'\ntrack all images {f_num}: {video_frame}')

            if not os.path.exists(os.path.join(output_root, anno_f)):
                os.mkdir(os.path.join(output_root, anno_f))

            txt_output_path = f'{output_root}/{anno_f}/{anno_n}_det.txt'

            file = open(txt_output_path, 'w')
            for image in tqdm(image_list):
                image_np = cv2.imread(os.path.join(image_path, image))
                num = image.split('.')[0]
                # class 정보는 안들어가서 뺌 -> class 정보 추가
                # box 여러개 확인
                boxes, classes, scores = detector.detect(image_np)
                for box, cls, score in zip(boxes, classes, scores):

                    #                         x0       y0     x1-x0            y1-y0
                    #          frame    id  bb-left  bb-top   bb-width         bb-height          conf   class
                    data = f'{int(num)},-1,{box[0]},{box[1]},{box[2] - box[0]},{box[3] - box[1]},{score},{cls}\n'
                    file.write(data)
            file.close()
    return


def get_image_list(image_path):
    image_list = os.listdir(image_path)
    image_list = sorted(image_list)

    # error check
    for i in range(0, len(image_list) - 1):
        i_idx = int(image_list[i].split('.')[0])
        next_i_idx = int(image_list[i + 1].split('.')[0])
        assert i_idx + 1 == next_i_idx

    return image_list


def main(args):
    anno_name = args.anno_name
    output_root = args.output_root
    image_root = args.image_root

    #write_txt_one(image_root, anno_name, output_root)
    write_txt_all(image_root, output_root)


if __name__ == '__main__':
    # 만들어질 파일 이름
    anno_name = '0001/2793806282'
    # 어떤 데이터셋
    mode = 'validation'
    # 어디다 저장
    output_root = '../../hdd_2tb/mot_neural_solver/data/vidor/labels/{}'.format(mode)
    # 이미지 폴더
    image_root = './vidor-dataset/frames_{}'.format(mode)

    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_name", type=str, default=anno_name)
    parser.add_argument("--output_root", type=str, default=output_root)
    parser.add_argument("--image_root", type=str, default=image_root)
    args = parser.parse_args()

    main(args)
