import os
import sys
import torch
sys.path.append(os.getcwd())
import cv2
import numpy as np
from tools.detect import MtcnnDetector, create_mtcnn_net
from tools.imagedb import ImageDB
from tools.image_reader import TestImageLoader
import time
from tools.tools import assemble_data

from six.moves import cPickle
from tools.tools import convert_to_square, IoU

prefix_path = "../DataSet/WIDER_train/images/"
traindata_store = '../Dataset/train/'
pnet_model_file = '../Model_store/pnet_epoch_10.pt'
rnet_model_file = '../Model_store/rnet_epoch_10.pt'
annotation_file = '../DataSet/my_anno.txt'
landmark_anno='../DataSet/my_landmark.txt'
save_path = '../Model_store'

onet_positive_file = '../DataSet/anno/pos_48.txt'
onet_part_file = '../DataSet/anno/part_48.txt'
onet_neg_file = '../DataSet/anno/neg_48.txt'
imglist_filename = '../DataSet/anno/imglist_anno_48.txt'

def gen_onet_data(data_dir, anno_file, pnet_model_file, rnet_model_file,landmark_anno, prefix_path='', device=torch.device('cpu')):
    pnet, rnet, _ = create_mtcnn_net(p_model_path=pnet_model_file, r_model_path=rnet_model_file, device=device)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, min_face_size=12,device=device)

    imagedb = ImageDB(anno_file, mode="test", prefix_path=prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb, 1, False)

    all_boxes = list()
    batch_idx = 0

    print('size:%d' % image_reader.size)
    for databatch in image_reader:
        if batch_idx % 50 == 0:
            print("%d images done" % batch_idx)

        im = databatch

        t = time.time()
        # pnet detection = [x1, y1, x2, y2, score, reg]
        p_boxes, p_boxes_align = mtcnn_detector.detect_pnet(im=im)
        t0 = time.time() - t
        t = time.time()
        # rnet detection
        boxes, boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)

        t1 = time.time() - t
        print('cost time pnet--', t0, '  rnet--', t1)

        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue


        all_boxes.append(boxes_align)
        batch_idx += 1


    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)


    gen_onet_sample_data(data_dir, anno_file, save_file, prefix_path,landmark_anno)


def gen_onet_sample_data(data_dir, anno_file, det_boxs_file, prefix,landmark_anno):
    neg_save_dir = os.path.join(data_dir, "48/negative")
    pos_save_dir = os.path.join(data_dir, "48/positive")
    part_save_dir = os.path.join(data_dir, "48/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    landmark={}
    with open(landmark_anno,'r') as f:
        landmark_annotations=f.readlines()
    for landmark_annotation in landmark_annotations:
        landmark_annotation=landmark_annotation.strip().split(' ')
        img=landmark_annotation[0]
        mark = list(map(float, landmark_annotation[1:]))
        mark=np.array(mark).reshape(-1,10)
        landmark[prefix_path+img]=mark


    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    with open(anno_file, 'r') as f:
        annotations = f.readlines()


    image_size = 48
    net = "onet"

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix, annotation[0])

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = './anno_store'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f1 = open(onet_positive_file, 'w')
    f2 = open(onet_neg_file, 'w')
    f3 = open(onet_part_file, 'w')

    det_handle = open(det_boxs_file, 'rb')

    det_boxes = cPickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    # assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1
        if gts.shape[0] == 0:
            continue
        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:              #候选框
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1


            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt      #真值
                mark=landmark[im_idx][idx]     #真值

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                landmark_offset_x1 = 0
                landmark_offset_y1 = 0
                landmark_offset_x2 = 0
                landmark_offset_y2 = 0
                landmark_offset_x3 = 0
                landmark_offset_y3 = 0
                landmark_offset_x4 = 0
                landmark_offset_y4 = 0
                landmark_offset_x5 = 0
                landmark_offset_y5 = 0
                if sum(mark)>0:
                    landmark_offset_x1=(mark[0]-x_left)/float(width)
                    landmark_offset_y1=(mark[1]-y_top)/float(height)
                    landmark_offset_x2=(mark[2]-x_left)/float(width)
                    landmark_offset_y2=(mark[3]-y_top)/float(height)
                    landmark_offset_x3=(mark[4]-x_left)/float(width)
                    landmark_offset_y3=(mark[5]-y_top)/float(height)
                    landmark_offset_x4=(mark[6]-x_left)/float(width)
                    landmark_offset_y4=(mark[7]-y_top)/float(height)
                    landmark_offset_x5=(mark[8]-x_left)/float(width)
                    landmark_offset_y5=(mark[9]-y_top)/float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2,landmark_offset_x1,landmark_offset_y1,landmark_offset_x2,
                        landmark_offset_y2,landmark_offset_x3,landmark_offset_y3,landmark_offset_x4,landmark_offset_y4,
                        landmark_offset_x5,landmark_offset_y5))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)

                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2,landmark_offset_x1,landmark_offset_y1,landmark_offset_x2,
                        landmark_offset_y2,landmark_offset_x3,landmark_offset_y3,landmark_offset_x4,landmark_offset_y4,
                        landmark_offset_x5,landmark_offset_y5))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_onet_data(traindata_store, annotation_file, pnet_model_file, rnet_model_file, landmark_anno,prefix_path, device)




    anno_list = []

    anno_list.append(onet_positive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)

    chose_count =assemble_data(imglist_filename ,anno_list)
    print("ONet train annotation result file path:%s" % imglist_filename)
