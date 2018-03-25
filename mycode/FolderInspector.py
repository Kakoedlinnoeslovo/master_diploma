from os import listdir
from os.path import isfile, join
import os


class FolderInspector:
    def __int__(self):
        pass

    def get_folder_list(self, path):
        return os.listdir(path)

    def get_files(self, path):
        onlyfiles = [f for f in listdir(path) if (isfile(join(path, f)))]
        return onlyfiles