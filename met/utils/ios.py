from pathlib import Path
from typing import Union


def mkdir_if_missing(dir_path: Union[Path, str]) -> None:
    directory = Path(dir_path)
    if directory.is_dir():
        return
    else:
        directory.mkdir(parents=True, exist_ok=True)


def delete_file(file_path):
    file_path.unlink(missing_ok=True)
