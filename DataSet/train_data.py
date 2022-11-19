import os
import numpy as np
import random


label='./label.txt'
save_name_anno='my_anno.txt'
save_name_lanmark='my_landmark.txt'
picture_num=2000     #提取图片的数量
thrd=0.2      #过滤阈值

with open(label,'r') as f:
    data=f.readlines()


img_data=[]
box=[0,1,2,3]
landmark=[4,5,7,8,10,11,13,14,16,17]
count=-1
for i,d in enumerate(data):
    if d.split(' ')[0] =='#':
        count+=1
        img_data.append({})
        img_data[count]['image']=d.split(' ')[1].strip()
        img_data[count]['box']=[]
        img_data[count]['landmark']=[]
        img_data[count]['score']=[]
    else:
        l=[]
        l2=[]
        bbox = list(map(float, d.strip().split()))
        for k in landmark:
            l.append(bbox[k])
        for k in box:
            l2.append(bbox[k])
        img_data[count]['box']+=l2
        img_data[count]['landmark']+=l
        img_data[count]['score'].append(bbox[19])
random.shuffle(img_data)

save_path=''
f1 = open(os.path.join(save_path,save_name_anno ), 'w')
f2 = open(os.path.join(save_path,save_name_lanmark),'w')
count=0
for t,d in enumerate(img_data):
    if (t+1)%200==0:
        print(f"{t+1} pirctures done")
    num=0
    box=np.array(d['box'])
    landmark=np.array(d['landmark'])
    score=np.array(d['score'])
    box=box.reshape(-1,4)
    landmark=landmark.reshape(-1,10)
    score=score.reshape(-1,1)
    use=[]
    for i in range(box.shape[0]):
        if sum(landmark[i])<0 or score[i]<thrd:
            continue
        else:
            use.append(i)
    if len(use)==0:
        continue
    f1.write(d['image'])
    f2.write(d['image'])
    for i in use:
        f1.write(' %d %d %d %d'%(int(box[i][0]),int(box[i][1]),int(box[i][0]+box[i][2]),int(box[i][1]+box[i][3])))
        f2.write(' '+str(landmark[i][0])+' '+str(landmark[i][1])+' '+str(landmark[i][2])+' '+str(landmark[i][3])+' '+str(landmark[i][4])+
                 ' '+str(landmark[i][5])+' '+str(landmark[i][6])+' '+str(landmark[i][7])+' '+str(landmark[i][8])+' '+str(landmark[i][9]))
    f1.write('\n')
    f2.write('\n')
    count+=1
    if count>picture_num:
        break

f1.close()
f2.close()