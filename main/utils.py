import os
import datetime
from config import VERSION_FILE

def get_version():
    # If the version file doesn't exist, create it with version 1
    if not VERSION_FILE.exists():
        with open(VERSION_FILE, 'w') as f:
            f.write("1")
        return 1

    # If the version file exists, read the current version
    with open(VERSION_FILE, 'r') as f:
        version = int(f.read())

    return version

def increment_version():
    # Get the current version
    version = get_version()

    # Increment the version
    version += 1

    # Write the new version to the file
    with open(VERSION_FILE, 'w') as f:
        f.write(str(version))

    return version


def folder(version_dir):
    """
    Checks if a directory exists at the given path, and if not, creates one.
    Also creates a log file in the new directory.
    """
    vdir = str(version_dir)
    if not version_dir.exists():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if "data" in vdir:
            os.makedirs(version_dir / "raw")
            os.makedirs(version_dir / "curated")

        elif "model" in version_dir:
            os.makedirs(version_dir)

        log = os.path.join(version_dir, "log.txt")

        with open(log, 'w') as f:
            f.write(f"Log file created : {timestamp} \n")

        print(f'Version Folder created at {version_dir}')

    else:
        print(f" VersionFolder {version_dir} already exists, skipping...")

