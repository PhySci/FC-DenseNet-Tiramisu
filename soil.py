import os
from utils import *
from FC_DenseNet_Tiramisu import Tiramisu

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    o = Tiramisu(preset_model='FC-DenseNet56', num_classes=2, img_size=[128, 128])
    #o.train(batch_size=4, num_epochs=1)
    o.predict('./soil/test/images')



