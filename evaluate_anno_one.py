import json
import argparse

from evaluation.dataset import VidOR
from evaluation import eval_video_object


def evaluate_object(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_object_insts(vid)
    mean_ap, ap_class = eval_video_object(groundtruth, prediction)
    return mean_ap, ap_class


if __name__ == '__main__':
    prediction_json_file = './outputs/eval_json/eval_tracklet.json'
    anno_path = './outputs/gt_json'
    #video_path = '/media/ailab/4f12b7e1-629d-424c-8fe5-35a9c2f8db76/jonghun/workspace/vidvrd/vidor-dataset/video'
    video_path = './vidor-dataset/video_validation'

    parser = argparse.ArgumentParser(description='Evaluate a set of tasks related to video relation understanding.')
    parser.add_argument('--dataset', choices=['imagenet-vidvrd', 'vidor'], help='the dataset name for evaluation', default='vidor')
    parser.add_argument('--split', choices=['validation', 'test'], help='the split name for evaluation', default='test')
    parser.add_argument('--task', choices=['object', 'relation'], help='which task to evaluate', default='object')
    parser.add_argument('--prediction', type=str, help='Corresponding prediction JSON file', default=prediction_json_file)
    args = parser.parse_args()

    if args.dataset=='vidor':
        if args.task=='object':
            # dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', [args.split], low_memory=True)
            dataset = VidOR(anno_path, video_path, [args.split], low_memory=True)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    print('Loading prediction from {}'.format(args.prediction))
    with open(args.prediction, 'rt') as fin:
        pred = json.load(fin)
    print('Number of videos in prediction: {}'.format(len(pred['results'])))

    if args.task=='object':
        evaluate_object(dataset, args.split, pred['results'])
