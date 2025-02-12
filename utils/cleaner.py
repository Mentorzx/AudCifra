import shutil
import sys
from pathlib import Path
from typing import Optional


def clean_logs_and_pycache(root_path: Optional[str], logs_dir: str = "logs") -> None:
    """
    Remove all .log files from the specified logs folder and delete all '__pycache__'
    directories recursively starting from the project root.

    If root_path is not provided, it is determined as the parent directory of the folder
    containing this script (assuming the script is in a subdirectory like 'utils').

    Parameters:
        root_path (str, optional): The root directory to search. If None, the project root is used.
        logs_dir (str): The name of the logs directory under the project root.
    """
    # Determine the project root if not provided.
    root = Path(root_path) if root_path else Path(__file__).resolve().parent.parent
    logs_path = root / logs_dir
    if logs_path.is_dir():
        for log_file in logs_path.glob("*.log"):
            try:
                log_file.unlink()
                print(f"Deleted log file: {log_file}")
            except Exception as e:
                print(f"Error deleting log file {log_file}: {e}", file=sys.stderr)
    else:
        print(f"Logs directory '{logs_path}' does not exist.", file=sys.stderr)

    for pycache_dir in root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            try:
                shutil.rmtree(pycache_dir)
                print(f"Deleted __pycache__ directory: {pycache_dir}")
            except Exception as e:
                print(
                    f"Error deleting __pycache__ directory {pycache_dir}: {e}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    clean_logs_and_pycache(None)
