import cv2
from tools.detect import create_mtcnn_net, MtcnnDetector
from tools.vision import vis_face
import torch
if __name__ == '__main__':

    file_path='mid.jpg'
    save_name = 'r_1.jpg'

    p_model_path = "../Model_store/pnet_epoch_10.pt"
    r_model_path = "../Model_store/rnet_epoch_10.pt"
    o_model_path = "../Model_store/onet_epoch_20.pt"

    # p_model_path = "../pre_trained/pnet_epoch_10.pt"
    # r_model_path = "../pre_trained/rnet_epoch_10.pt"
    # o_model_path = "../pre_trained/onet_epoch_20.pt"

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model_path, r_model_path=r_model_path, o_model_path=o_model_path,
                                        device=device)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24, threshold=[0.3, 0.3, 0.5],device=device)

    img = cv2.imread(file_path)
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxs, landmarks = mtcnn_detector.detect_face(img)
    vis_face(img_bg, bboxs, landmarks, save_name)
