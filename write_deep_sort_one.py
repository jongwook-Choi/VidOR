import argparse
import os
from annotation import annotation_model


def main(args, mode):
    model = annotation_model.AnnotationModel(args)
    annotation_model.write_one_json(model, args, mode)


if __name__ == '__main__':
    #anno_name = '1111/2892594713.json'
    anno_name = '0033/6068085283.json'
    mode = 'training'

    output_root = './outputs/output_json'
    #image_root = '/media/ailab/4f12b7e1-629d-424c-8fe5-35a9c2f8db76/jonghun/workspace/vidvrd/vidor-dataset/frame'
    image_root = './vidor-dataset/frames_training'
    #anno_root = '/media/ailab/4f12b7e1-629d-424c-8fe5-35a9c2f8db76/jonghun/workspace/vidvrd/vidor-dataset'
    anno_root = './vidor-dataset'

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default=output_root)
    parser.add_argument("--image_root", type=str, default=image_root)
    parser.add_argument("--anno_root", type=str, default=anno_root)
    parser.add_argument("--anno_name", type=str, default=anno_name)
    args = parser.parse_args()

    main(args, mode)
