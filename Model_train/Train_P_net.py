import sys
import os

import torch

sys.path.append(os.getcwd())
from tools.imagedb import ImageDB
from train import train_pnet

annotation_file='../DataSet/anno/imglist_anno_12.txt'
model_store_path = '../Model_store/'
end_epoch = 10
frequent = 100
lr = 0.01
lr_epoch_decay = [9]
batch_size = 640
def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01,lr_epoch_decay=[9],
                 batch_size=128, device=torch.device('cpu')):

    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    print('DATASIZE',len(gt_imdb))
    gt_imdb = imagedb.append_flipped_images(gt_imdb)
    print('FLIP DATASIZE',len(gt_imdb))

    train_pnet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb, batch_size=batch_size, frequent=frequent, base_lr=lr,lr_epoch_decay=lr_epoch_decay,device=device)

if __name__ == '__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_net(annotation_file, model_store_path,
                end_epoch, frequent, lr, lr_epoch_decay,batch_size,device)
