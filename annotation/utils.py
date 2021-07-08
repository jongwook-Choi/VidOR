

def get_category_name_dict(file_path='./configs/vidor_object.txt'):
    object_dict = {}

    with open(file_path) as vidor_file:
        while True:
            line = vidor_file.readline()

            if not line:
                break

            category_name, category_id = line.split()
            object_dict[int(category_id)] = category_name

    return object_dict


def get_category_id_dict(file_path='./configs/vidor_object.txt'):
    object_dict = {}

    with open(file_path) as vidor_file:
        while True:
            line = vidor_file.readline()

            if not line:
                break

            category_name, category_id = line.split()
            object_dict[category_name] = int(category_id)

    return object_dict


