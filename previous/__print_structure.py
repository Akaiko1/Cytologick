import os

IGNORE_FOLDERS=['.git', '__pycache__', 'slide-', 'ai', 'temp', 'static']
IGNORE_FILES = ['slide']


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        if any(f in root for f in IGNORE_FOLDERS):
            continue
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


if __name__ == '__main__':
    list_files('./current')
