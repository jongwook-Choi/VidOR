import argparse
import os
from annotation import annotation_model


def main(args, mode):
    model = annotation_model.AnnotationModel(args)
    annotation_model.write_all_json(model, args, mode)


if __name__ == '__main__':
    #anno_name = '1111/2892594713.json'
    # train or test
    mode = 'validation'

    output_root = './outputs/output_json'
    image_root = './vidor-dataset/frames_validation'
    anno_root = './vidor-dataset'

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default=output_root)
    parser.add_argument("--image_root", type=str, default=image_root)
    parser.add_argument("--anno_root", type=str, default=anno_root)
    args = parser.parse_args()

    main(args, mode)
