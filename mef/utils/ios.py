import pickle
import sys
from pathlib import Path


def mkdir_if_missing(dir_path):
    directory = Path(dir_path)
    if directory.is_dir():
        return
    else:
        directory.mkdir(parents=True, exist_ok=True)


def generate_token(x):
    names = []
    for key in x:
        category = key
        value = x[key]

        if type(value) == bool:
            if value:
                name = category
            else:
                continue
        elif isinstance(value, type(None)):
            name = category + '=' + "None"
        elif isinstance(value, int):
            name = category + '=' + str(value)
        elif isinstance(value, float):
            name = category + '=' + str(value)
        elif isinstance(value, str):
            name = category + '=' + value
        elif isinstance(value, bytes):
            name = category + '=' + value
        elif isinstance(value, list):
            name = category + '=' + str(value)
        elif isinstance(value, dict):
            name = category + '=' + '(' + generate_token(value) + ')'
        else:
            raise NotImplementedError(
                "generate_folder_name: take care of the case for type " +
                str(type(value)))

        names.append(name)

    names.sort()
    return ",".join(names)


def delete_file(file_path):
    file_path.unlink(missing_ok=True)


def save_to_file(content, file_path):
    mkdir_if_missing(file_path.parent)
    with open(file_path, "wb") as fp:
        pickle.dump(content, fp)


def load_file(file_path, python23_conversion=False):
    fp = open(file_path, "rb")
    if (sys.version_info[0] >= 3) and python23_conversion:
        return pickle.load(fp, encoding='latin1')
    return pickle.load(fp)
