from eval_track import test_anno

if __name__ == "__main__":
    json_root = './outputs/output_json'
    #json_name = '1111/2892594713.json'
    json_name = '0033/6068085283.json'
    output_root = './outputs/eval_json'

    test_anno.write_anno(json_root, json_name, output_root)

