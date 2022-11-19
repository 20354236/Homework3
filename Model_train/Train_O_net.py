import os
import sys
sys.path.append(os.getcwd())
from tools.imagedb import ImageDB
import train as train
import torch

def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01,lr_epoch_decay=[], batch_size=128,  device=torch.device('cpu')):

    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)

    train.train_onet(model_store_path=model_store_path, end_epoch=end_epoch,
        imdb=gt_imdb, batch_size=batch_size, frequent=frequent, lr_epoch_decay=lr_epoch_decay,
        base_lr=lr,  device=device)

if __name__ == '__main__':

    print('train ONet argument:')

    annotation_file = "../DataSet/anno/imglist_anno_48.txt"
    model_store_path = "../Model_store"
    end_epoch = 20
    lr = 0.005
    batch_size = 640


    frequent = 50
    lr_epoch_decay = [8]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_net(annotation_file, model_store_path,
                end_epoch, frequent, lr,lr_epoch_decay, batch_size, device)
