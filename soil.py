import os
from FC_DenseNet_Tiramisu import Tiramisu


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    o = Tiramisu(preset_model='FC-DenseNet103', num_classes=2, img_size=[128, 128])
    o.train(batch_size=8, num_epochs=250)
    o.predict(test_pth='./soil/test/images', save_pth='./predictions')
