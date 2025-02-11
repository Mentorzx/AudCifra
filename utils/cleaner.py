import os
import shutil


def clean_logs_and_pycache(root_path: str = ".", logs_dir: str = "logs") -> None:
    """
    Removes all .log files from the specified logs folder
    and deletes any '__pycache__' directories under root_path.

    :param root_path: The root directory to search (usually the project root).
    :param logs_dir: The directory where log files are stored.
    """
    logs_path = os.path.join(root_path, logs_dir)
    if os.path.isdir(logs_path):
        for f in os.listdir(logs_path):
            if f.endswith(".log"):
                os.remove(os.path.join(logs_path, f))

    for dirpath, dirnames, filenames in os.walk(root_path):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(pycache_path)


if __name__ == "__main__":
    clean_logs_and_pycache()
