import datetime
import numpy as np
import torch
import os
from Model.Loss import LossFn
from Model.model import P_Net,R_Net,O_Net
from torch.autograd import Variable
from tools.image_reader import TrainImageReader
import tools.image_tools as image_tools
from tools.tools import compute_accuracy
import matplotlib.pyplot as plt
def train_pnet(model_store_path, end_epoch, imdb,
               batch_size, frequent=10, base_lr=0.01, lr_epoch_decay=[9], device=torch.device('cpu')):

    bbox_loss_list=[]
    cls_loss_list=[]
    acc_list=[]
    # create lr_list
    lr_epoch_decay.append(end_epoch + 1)
    lr_list = np.zeros(end_epoch)
    lr_t = base_lr
    for i in range(len(lr_epoch_decay)):
        if i == 0:
            lr_list[0:lr_epoch_decay[i] - 1] = lr_t
        else:
            lr_list[lr_epoch_decay[i - 1] - 1:lr_epoch_decay[i] - 1] = lr_t
        lr_t *= 0.1

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = P_Net()
    net.train()
    net=net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr_list[0])

    train_data = TrainImageReader(imdb, 12, batch_size, shuffle=True)

    # frequent = 10
    for cur_epoch in range(1, end_epoch + 1):
        train_data.reset()  # shuffle
        for param in optimizer.param_groups:
            param['lr'] = lr_list[cur_epoch - 1]
        for batch_idx, (image, (gt_label, gt_bbox, _)) in enumerate(train_data):

            im_tensor = [image_tools.convert_image_to_tensor(image[i, :, :, :]) for i in range(image.shape[0])]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())

            gt_label=gt_label.to(device)
            gt_bbox=gt_bbox.to(device)
            im_tensor=im_tensor.to(device)

            cls_pred, box_offset_pred = net(im_tensor)

            cls_loss = lossfn.cls_loss(gt_label, cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)

            cls_loss_list.append(float(cls_loss.cpu().detach().numpy()))
            bbox_loss_list.append(float(box_offset_loss.cpu().detach().numpy()))



            all_loss = cls_loss * 1.0 + box_offset_loss * 0.5

            if batch_idx % frequent == 0:
                accuracy = compute_accuracy(cls_pred, gt_label)
                acc_list.append(float(accuracy.cpu().numpy()))

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s " % (
                datetime.datetime.now(), cur_epoch, batch_idx, show1, show2, show3, show5, lr_list[cur_epoch - 1]))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        if cur_epoch%10==0:
            torch.save(net.state_dict(), os.path.join(model_store_path, "pnet_epoch_%d.pt" % cur_epoch))
            torch.save(net, os.path.join(model_store_path, "pnet_epoch_model_%d.pkl" % cur_epoch))
    x_loss=[i for i in range(len(bbox_loss_list))]
    x_acc=[i for i in range(len(acc_list))]
    plt.subplot(311)
    plt.plot(x_loss,cls_loss_list)
    plt.ylabel("cls_loss")

    plt.subplot(312)
    plt.plot(x_loss,bbox_loss_list)
    plt.ylabel("bbox_loss")

    plt.subplot(313)
    plt.plot(x_acc,acc_list)
    plt.ylabel("accuary")
    plt.savefig("Pnet.jpg")
    plt.show()



def train_rnet(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,lr_epoch_decay=[9],device=torch.device('cpu')):

    bbox_loss_list=[]
    cls_loss_list=[]
    acc_list=[]

    #create lr_list
    lr_epoch_decay.append(end_epoch+1)
    lr_list = np.zeros(end_epoch)
    lr_t = base_lr
    for i in range(len(lr_epoch_decay)):
        if i==0:
            lr_list[0:lr_epoch_decay[i]-1]=lr_t
        else:
            lr_list[lr_epoch_decay[i-1]-1:lr_epoch_decay[i]-1]=lr_t
        lr_t*=0.1

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = R_Net()
    net.train()
    net=net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,24,batch_size,shuffle=True)

    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()
        for param in optimizer.param_groups:
            param['lr'] = lr_list[cur_epoch-1]

        for batch_idx,(image,(gt_label,gt_bbox,_))in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())

            gt_label=gt_label.to(device)
            gt_bbox=gt_bbox.to(device)
            im_tensor=im_tensor.to(device)

            cls_pred, box_offset_pred= net(im_tensor)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)

            cls_loss_list.append(float(cls_loss.cpu().detach().numpy()))
            bbox_loss_list.append(float(box_offset_loss.cpu().detach().numpy()))

            all_loss = cls_loss*1.0+box_offset_loss*0.5

            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)
                acc_list.append(float(accuracy.cpu().numpy()))


                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(), cur_epoch, batch_idx, show1, show2, show3, show5, lr_list[cur_epoch-1]))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        if cur_epoch%10==0:
            torch.save(net.state_dict(), os.path.join(model_store_path,"rnet_epoch_%d.pt" % cur_epoch))
            torch.save(net, os.path.join(model_store_path,"rnet_epoch_model_%d.pkl" % cur_epoch))
    x_loss=[i for i in range(len(bbox_loss_list))]
    x_acc=[i for i in range(len(acc_list))]
    plt.subplot(311)
    plt.plot(x_loss,cls_loss_list)
    plt.ylabel("cls_loss")

    plt.subplot(312)
    plt.plot(x_loss,bbox_loss_list)
    plt.ylabel("bbox_loss")

    plt.subplot(313)
    plt.plot(x_acc,acc_list)
    plt.ylabel("accuary")
    plt.savefig("Rnet.jpg")
    plt.show()



def train_onet(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,lr_epoch_decay=[9],device=torch.device('cpu')):

    bbox_loss_list=[]
    cls_loss_list=[]
    acc_list=[]
    landmark_loss_list=[]
    #create lr_list
    lr_epoch_decay.append(end_epoch+1)
    lr_list = np.zeros(end_epoch)
    lr_t = base_lr
    for i in range(len(lr_epoch_decay)):
        if i==0:
            lr_list[0:lr_epoch_decay[i]-1]=lr_t
        else:
            lr_list[lr_epoch_decay[i-1]-1:lr_epoch_decay[i]-1]=lr_t
        lr_t*=0.1

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = O_Net()
    net.train()
    net=net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,48,batch_size,shuffle=True)


    for cur_epoch in range(1,end_epoch+1):

        train_data.reset()
        for param in optimizer.param_groups:
            param['lr'] = lr_list[cur_epoch-1]
        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):
            #print("batch id {0}".format(batch_idx))
            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            gt_label=gt_label.to(device)
            gt_bbox=gt_bbox.to(device)
            im_tensor=im_tensor.to(device)
            gt_landmark=gt_landmark.to(device)

            cls_pred, box_offset_pred, landmark_offset_pred = net(im_tensor)


            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)


            cls_loss_list.append(float(cls_loss.cpu().detach().numpy()))
            bbox_loss_list.append(float(box_offset_loss.cpu().detach().numpy()))
            landmark_loss_list.append(float(landmark_loss.cpu().detach().numpy()))

            all_loss = cls_loss*0.8+box_offset_loss*0.6+landmark_loss*1.5


            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)
                acc_list.append(float(accuracy.cpu().numpy()))

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s,\n det loss: %s, bbox loss: %s, landmark loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show4,show5,base_lr))
                #print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show5,lr_list[cur_epoch-1]))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        if cur_epoch%10==0:
            torch.save(net.state_dict(), os.path.join(model_store_path,"onet_epoch_%d.pt" % cur_epoch))
            torch.save(net, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % cur_epoch))
    x_loss=[i for i in range(len(bbox_loss_list))]
    x_acc=[i for i in range(len(acc_list))]
    plt.subplot(411)
    plt.plot(x_loss,cls_loss_list)
    plt.ylabel("cls_loss")

    plt.subplot(412)
    plt.plot(x_loss,bbox_loss_list)
    plt.ylabel("bbox_loss")

    plt.subplot(413)
    plt.plot(x_loss,landmark_loss_list)
    plt.ylabel('landmark_loss')

    plt.subplot(414)
    plt.plot(x_acc,acc_list)
    plt.ylabel("accuary")
    plt.savefig("Onet.jpg")
    plt.show()
