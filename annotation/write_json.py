import json
from tqdm import tqdm
from annotation.utils import get_category_name_dict


def check_error(frame_count, total_video_track_list, subject_object_dict_list, total_trajectories):
    assert frame_count == len(total_video_track_list)
    assert frame_count == len(total_trajectories)


def get_anno_info(anno_path):
    with open(anno_path) as anno_name:
        json_data = json.load(anno_name)

        width = json_data['width']
        frame_count = json_data['frame_count'] #
        video_id = json_data['video_id']
        fps = json_data['fps']
        height = json_data['height']

    return width, frame_count, video_id, fps, height


def get_subject_objects(total_video_track_list):
    """
    :param total_video_track_list: [video_track_list, ...]
    video_track_list = [ [xmin, ymin, xmax, ymax, trajectory_id, class_id], ... ]

    :return: subject_object_dict_list
    subject_object_dict = {"category": "car", "tid": 0}
    """
    subject_object_dict = {}
    category_name_dict = get_category_name_dict()

    for video_track_list in tqdm(total_video_track_list):
        for video_track in video_track_list:
            [_, _, _, _, trajectory_id, class_id] = video_track

            if trajectory_id not in subject_object_dict.keys():
                subject_object_dict[trajectory_id] = category_name_dict[class_id]

    subject_object_dict_list = []

    for tid in subject_object_dict.keys():
        category = subject_object_dict[tid]
        dictionary = {"category": category, "tid": tid}
        subject_object_dict_list.append(dictionary)

    return subject_object_dict_list


def get_trajectories(total_video_track_list):
    """
    :param total_video_track_list: [video_track_list, ...]
    video_track_list = [ [xmin, ymin, xmax, ymax, trajectory_id, class_id], ... ]

    :return: total_trajectories: [
        [ video_track_dict, ...],
        ...
    ]
    """
    total_trajectories = []

    for video_track_list in tqdm(total_video_track_list):
        trajectories = []

        for video_track in video_track_list:
            [xmin, ymin, xmax, ymax, trajectory_id, _] = video_track
            video_track_dict = {
                "tid": trajectory_id,
                "generated": 1,
                "bbox": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
            }
            trajectories.append(video_track_dict)

        total_trajectories.append(trajectories)

    return total_trajectories

