import os
from random import shuffle
import shutil

def main(train_dir='', val_dir='', val_size=500):

    file_list = os.listdir(os.path.join(train_dir, 'images'))
    shuffle(file_list)
    val_files = file_list[:val_size]
    print(val_files)
    print(len(val_files))

    for file_name in val_files:
        shutil.move(os.path.join(train_dir, 'images', file_name),
                    os.path.join(val_dir, 'images', file_name))
        shutil.move(os.path.join(train_dir, 'masks', file_name),
                    os.path.join(val_dir, 'masks', file_name))


if __name__ == '__main__':
    main(train_dir='./soil/train', val_dir='./soil/val')