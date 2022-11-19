import os
import numpy as np
import cv2
import tools.tools as tools
from tools.tools import assemble_data


anno_file='../DataSet/my_anno.txt'
image_dir='../DataSet/WIDER_train/images/'
pos_save_dir='../DataSet/train/12/positive/'
part_save_dir = "../DataSet/train/12/part/"
neg_save_dir = '../DataSet/train/12/negative/'
pos_save_anno='../DataSet/anno/pos_12.txt'
neg_save_anno='../DataSet/anno/neg_12.txt'
part_save_anno='../DataSet/anno/part_12.txt'
imglist_filename = '../DataSet/anno/imglist_anno_12.txt'


NUM_Neg=5
NUM_Neg_Around=2
NUM_Positive_Part=5
print("生成PNet训练数据")
if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.makedirs(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir)


f1 = open(pos_save_anno, 'w')
f2 = open(neg_save_anno, 'w')
f3 = open(part_save_anno, 'w')

with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)


pos_idx=0
neg_idx=0
d_idx=0
idx=0
count=0

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    img_path = os.path.join(image_dir, annotation[0])
    bbox = list(map(float, annotation[1:]))
    boxes=np.array(bbox,dtype=np.int32).reshape(-1,4)
    if boxes.shape[0]==0:
        continue
    img=cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # b,g,r=cv2.split(img)
    # img=cv2.merge([r,g,b])
    # plt.imshow(img)
    # plt.show()
    idx +=1

    height, width, channel=img.shape
    neg_num=0

    while neg_num<NUM_Neg:               #随机选取 negtive 的区域
        size=np.random.randint(12,min(width,height)/2)
        nx=np.random.randint(0,width-size)
        ny=np.random.randint(0,height-size)
        crop_box=np.array([nx,ny,nx+size,ny+size])

        iou=tools.IoU(crop_box,boxes)

        cropped_im=img[ny:ny+size,nx:nx+size,:]
        resized_im=cv2.resize(cropped_im,(12,12),interpolation=cv2.INTER_LINEAR)

        if np.max(iou)<0.3:
            #negtive
            save_file = os.path.join(neg_save_dir, "%s.jpg" % neg_idx)
            f2.write(save_file+' 0\n')
            cv2.imwrite(save_file,resized_im)
            neg_num+=1
            neg_idx+=1
    for box in boxes:
        x1,y1,x2,y2=box
        w=x2-x1+1
        h=y2-y1+1

        if max(w,h)<30 or x1<0 or y1<0:
            continue
        if w<12 or h<12:
            continue
        for i in range(NUM_Neg_Around):                #选取 ground truth 周围的 negtive 的区域
            size=np.random.randint(12,min(width,height)/2)

            #offsets
            delta_x=np.random.randint(max(-size,-x1),w)
            delta_y=np.random.randint(max(-size,-y1),h)

            nx1=max(0,x1+delta_x)
            ny1=max(0,y1+delta_y)

            if nx1+size>width or ny1+size>height:
                continue
            crop_box=np.array([nx1,ny1,nx1+size,ny1+size])
            iou=tools.IoU(crop_box,boxes)

            cropped_im=img[ny1:ny1+size,nx1:nx1+size,:]
            resized_im=cv2.resize(cropped_im,(12,12),interpolation=cv2.INTER_LINEAR)

            if np.max(iou)<0.3:
                # negtive
                save_file = neg_save_dir + str(neg_idx) + '.jpg'
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                neg_idx += 1

        for i in range(NUM_Positive_Part):    #positive or part
            size=np.random.randint(int(min(w,h)*0.8),np.ceil(1.25*max(w,h)))

            delta_x=np.random.randint(-w*0.2,w*0.2)
            delta_y=np.random.randint(-h*0.2,h*0.2)

            nx1=max(0,x1+w/2+delta_x-size/2)
            ny1=max(0,y1+h/2+delta_y-size/2)
            nx2=nx1+size
            ny2=ny1+size

            if nx2>width or ny2>height:
                continue

            crop_box=np.array([nx1,ny1,nx2,ny2])

            offset_x1=(x1-nx1)/float(size)
            offset_y1=(y1-ny1)/float(size)
            offset_x2=(x2-nx2)/float(size)
            offset_y2=(y2-ny2)/float(size)

            cropped_im=img[int(ny1):int(ny2),int(nx1):int(nx2),:]
            resized_im=cv2.resize(cropped_im,(12,12), interpolation=cv2.INTER_LINEAR)

            box_=box.reshape(1,-1)
            if tools.IoU(crop_box,box_)>=0.65:
                save_file=pos_save_dir+str(pos_idx)+'.jpg'
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file,resized_im)
                pos_idx+=1
            elif tools.IoU(crop_box,box_)>=0.4:
                save_file=part_save_dir+str(d_idx)+'.jpg'
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file,resized_im)
                d_idx+=1
    if (count+1)%10==0:
        print("%s images done, pos: %s part: %s neg: %s" % (count+1, pos_idx, d_idx, neg_idx))
    count+=1

f1.close()
f2.close()
f3.close()

print("1")





anno_list = []

anno_list.append(pos_save_anno)
anno_list.append(part_save_anno)
anno_list.append(neg_save_anno)

chose_count = assemble_data(imglist_filename, anno_list)
print("PNet train annotation result file path:%s" % imglist_filename)