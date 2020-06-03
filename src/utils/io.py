import cv2
import ntpath
import os


def load_image(f, flag=cv2.IMREAD_COLOR):
    image = cv2.imread(f, flag)
    if flag == cv2.IMREAD_COLOR:
        return image[:, :, ::-1]  # Change from BGR to RGB
    else:
        return image


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_most_recent_file(dirpath):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dirpath)) for f in fn]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

