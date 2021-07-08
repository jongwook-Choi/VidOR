import json
import lzma
import os
from tqdm import tqdm

def get_anno_data(json_root, json_name):
    json_path = os.path.join(json_root, json_name)

    with open(json_path) as file:
        json_data = json.load(file)
        verson = "VERSION 1.0"
        video_id = json_data["video_id"]
        trajectories = json_data["trajectories"]
        subject_objects = json_data["subject/objects"]
        frame_count = json_data["frame_count"]

    return verson, video_id, trajectories, subject_objects, frame_count


def get_track_eval_format(trajectories, subject_objects):
    result_list = []
    tid_dict = {}
    so_name_dict = {}

    '''
    tid_dict = {
        0(tid): {
            "0"(frame_i) : [xmin, ymin, xmax, ymax],
            ...
        },
        ...
    }
    '''
    for so in subject_objects:
        tid = so["tid"]
        tid_dict[tid] = {}
        so_name_dict[tid] = so["category"]

    for frame_i, traj_list in enumerate(trajectories):
        for traj in traj_list:
            tid = traj["tid"]
            bbox = traj["bbox"]
            xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
            tid_dict[tid][str(frame_i)] = [xmin, ymin, xmax, ymax]

    for tid in tid_dict.keys():
        category = so_name_dict[tid]
        score = 0.99
        traj = tid_dict[tid]

        result_dict = {
            "category": category,
            "score": score,
            "trajectory": traj
        }

        result_list.append(result_dict)

    return result_list


def get_external_info():
    external_dict = {
        "used": True,
        "details": "First fully-connected layer from VGG-16 pre-trained on ILSVRC-2012 training set"
    }

    return external_dict


def write_anno(json_root, json_name, output_dir):
    verson, video_id, trajectories, subject_objects, frame_count = get_anno_data(json_root, json_name)
    result_list = get_track_eval_format(trajectories, subject_objects)
    external_dict = get_external_info()

    data_dict = {
        "version": verson,
        "results": {},
        "external_data": external_dict
    }

    data_dict["results"][video_id] = result_list
    json_output_path = os.path.join(output_dir, "eval_tracklet.json")

    with open(json_output_path, 'w') as fout:
        json.dump(data_dict, fout)

def write_anno_all(json_root, output_dir) :
    init = True
    json_folder_list = os.listdir(json_root)
    print("Start loop folder!")
    for json_folder in json_folder_list:
        json_folder_path = os.path.join(json_root, json_folder)
        json_list = os.listdir(json_folder_path)
        for json_file in json_list:
            json_name = "{}/{}".format(json_folder, json_file)
            verson, video_id, trajectories, subject_objects, frame_count = get_anno_data(json_root, json_name)
            result_list = get_track_eval_format(trajectories, subject_objects)
            external_dict = get_external_info()
            if init :
                data_dict = {
                    "version": verson,
                    "results": {},
                    "external_data": external_dict
                }
                data_dict["results"][video_id] = result_list
                init = False
            else :
                if data_dict["version"] != verson:
                    raise Exception("version error")
                if data_dict["external_data"] != external_dict:
                    raise Exception("external_data error")
                data_dict["results"][video_id] = result_list

    print("Start make .xz!")
    filename = "eval_tracklet.json.xz"
    output_path = os.path.join(output_dir, filename)
    with lzma.open(output_path, 'wt', check=lzma.CHECK_NONE) as fp:
        json.dump(data_dict, fp=fp, separators=(',', ':')) # separators : 구분자 설정
    print("Finish!")