from eval_track import test_anno

if __name__ == "__main__":
    json_root = './outputs/output_json/validation'
    output_root = './outputs/eval_json'

    test_anno.write_anno_all(json_root, output_root)