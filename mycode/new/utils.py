import os
from os import listdir
from os.path import isfile, join

class FolderViewer:
    def __init__(self):
        pass

    def get_folder_list(self, path):
        return os.listdir(path)

    def get_files(self, path, format = 'jpg'):
        onlyfiles = [path + f for f in listdir(path) if (isfile(join(path, f))) and f.split('.')[-1] == format]
        return onlyfiles

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


