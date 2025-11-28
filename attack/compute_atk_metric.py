from os import PathLike
import os
import numpy as np
import pickle as pkl
import numba
from utils.kitti import kitti_utils

import argparse  
from pathlib import Path  
from tqdm import tqdm
from shapely.geometry import LineString, box 
from shapely import affinity

# borrow from: https://blog.csdn.net/SpriteNym/article/details/127965618
def compute_iou_3d(gt_bbox, dt_bbox):
    """
    box{1,2}: [1, 7], [x, y, z, l, w, h, ry]
    """
    result_xy, result_z, result_v = [], [], []
    for b in [gt_bbox, dt_bbox]:
        x, y, z, l, w, h, yaw = b
        
        result_v.append(l * w * h)
        
        ls = LineString([[0, z - h/2], [0, z + h/2]])
        result_z.append(ls)
        
        poly = box(x - l/2, y - w/2, x + l/2, y + w/2)
        poly_rot = affinity.rotate(poly, yaw, use_radians=True)
        result_xy.append(poly_rot)

    overlap_xy = result_xy[0].intersection(result_xy[1]).area
    overlap_z = result_z[0].intersection(result_z[1]).length
    
    overlap_xyz = overlap_z * overlap_xy
    return overlap_xyz / (np.sum(result_v) - overlap_xyz)


def compute_atk_success_rate(
    label_dir:   PathLike, 
    calib_dir:   PathLike,
    result_file: PathLike, 
    iou_thred=0.7, 
    model_name=''
):
    
    gt_label_list = os.listdir(label_dir)
    gt_label_list.sort()
    calib_list = os.listdir(calib_dir)
    calib_list.sort()

    # NOTE: 25-05-18: gt_label 是按 3 帧生成的，而现在烟雾只有一帧，需要对齐
    new_gt_label_list = []
    offset = 0
    for calib_file in calib_list: 
        while True:
            gt_label_file = gt_label_list[offset]
            if (calib_file == gt_label_file):
                new_gt_label_list.append(gt_label_file)
                break
            else:
                offset += 1
                continue
    gt_label_list = new_gt_label_list

    with open(result_file, 'rb') as f:
        pred_result = pkl.load(f)

    num_total_label_car = 0
    num_total_pred_true = 0
    #for i in range(30):
    # for i in range(len(gt_label_list)):
    # for i in tqdm(range(len(pred_result)), desc='attack ' + model_name):
    pbar = tqdm(range(len(pred_result)), desc='ATK ' + model_name)
    for i, element in enumerate(pbar):
        # =====
        #print(f"in file: {gt_label_list[i]}")
        ## =====
        if (i % 1000 == 0):
            print("")
        gt_bbox_list, gt_type_list = kitti_utils.kitti_read_bbox(os.path.join(label_dir, gt_label_list[i]))
        dt_bbox_list, dt_type_list = pred_result[i]['pred_boxes'], pred_result[i]['pred_labels']
        
        # NOTE:
        score = pred_result[i]['pred_scores']
        temp_list = [dt_bbox_list[k] for k in range(len(score)) if score[k] > 0.5]
        dt_bbox_list = temp_list
        temp_list = [dt_type_list[k] for k in range(len(score)) if score[k] > 0.5]
        dt_type_list = temp_list

        gt_car_bbox_list = [gt_bbox_list[k] for k in range(len(gt_type_list)) if gt_type_list[k] == 'Car' or gt_type_list[k] == 'car']
        dt_car_bbox_list = [dt_bbox_list[k] for k in range(len(dt_type_list)) if dt_type_list[k] == 1]


        num_car = len(gt_car_bbox_list)
        if num_car == 0:
            continue
        num_pred_true = 0

        assert(gt_label_list[i] == calib_list[i])
        kitti_utils.init(os.path.join(calib_dir, calib_list[i]))
        # TODO: 并行计算 ?
        for gt_bbox in gt_car_bbox_list:
            x, y, z, l, h, w, ry  = gt_bbox
            center_in_camera  = np.array([[x, y, z]], dtype=np.float32)
            center_in_lidar   = kitti_utils.camera_to_lidar(center_in_camera)
            location_in_lidar = center_in_lidar + [0, 0, h/2]
            x, y, z = location_in_lidar[0]
            # borrow from openpcdet
            # (N, 7) [x, y, z, dx, dy, dz, heading] 
            # -(np.pi / 2 + ry) 因为在kitti中，camera坐标系下定义物体朝向与camera的x轴夹角顺时针为正，逆时针为负
            # 在pcdet中，lidar坐标系下定义物体朝向与lidar的x轴夹角逆时针为正，顺时针为负，所以二者本身就正负相反
            # pi / 2是坐标系x轴相差的角度(如图所示)
            # camera:         lidar:
            # Y                    X
            # |                    |
            # |____X         Y_____|  
            gt_bbox = [x, y, z, l, w, h, -(np.pi / 2 + ry)]

            for dt_bbox in dt_car_bbox_list:
                #print("iou: ", compute_iou_3d(gt_bbox, dt_bbox))
                if compute_iou_3d(gt_bbox, dt_bbox) > iou_thred:
                    num_pred_true += 1
                else:
                    continue
        num_total_pred_true += num_pred_true
        num_total_label_car += num_car

        atk_success_rate = (num_car - num_pred_true) / num_car
        #print("total car: ", num_total_label_car)
        #print("total pred: ", num_total_pred_true)

        # print(atk_success_rate)
        total_atk_success_rate = (num_total_label_car - num_total_pred_true) / num_total_label_car
        pbar.set_postfix({"total_asr": total_atk_success_rate})        
    total_atk_success_rate = (num_total_label_car - num_total_pred_true) / num_total_label_car
    print(f"total_atk_rate: {total_atk_success_rate}")
    # print(f"total_det_rate: {1 - total_atk_success_rate}")
    return None

def nusc_compute_atk_success_rate(
    label_dir:   PathLike, 
    result_file: PathLike, 
    iou_thred=0.5,
    model_name=''
):
    gt_label_list = os.listdir(label_dir)
    gt_label_list.sort()
    with open(result_file, 'rb') as f:
        pred_result = pkl.load(f)

    # # NOTE: 25-05-18: gt_label 是按 3 帧生成的，而现在烟雾只有一帧，需要对齐
    new_gt_label_list = []
    for gt_label_name in gt_label_list:
        if gt_label_name.endswith('_0.pkl'):
            new_gt_label_list.append(gt_label_name)
    gt_label_list = new_gt_label_list

    num_total_label_car = 0
    num_total_pred_true = 0
    #for i in range(30):
    # for i in range(len(pred_result)):
    pbar = tqdm(range(len(pred_result)), desc='ATK ' + model_name)
    for i, element in enumerate(pbar):
        # =====
        #print(f"in file: {gt_label_list[i]}")
        ## =====
        gt_label_file = os.path.join(label_dir, gt_label_list[i])
        with open(gt_label_file, 'rb') as f:
            gt_car_bbox_list = pkl.load(f)
            
        dt_bbox_list, dt_type_list = pred_result[i]['pred_boxes'], pred_result[i]['pred_labels']
        
        # NOTE:
        score = pred_result[i]['pred_scores']
        temp_list = [dt_bbox_list[k] for k in range(len(score)) if score[k] > 0.5]
        dt_bbox_list = temp_list
        temp_list = [dt_type_list[k] for k in range(len(score)) if score[k] > 0.5]
        dt_type_list = temp_list

        dt_car_bbox_list = [dt_bbox_list[k] for k in range(len(dt_type_list)) if dt_type_list[k] == 1]

        num_car = len(gt_car_bbox_list)
        if num_car == 0:
            continue
        num_pred_true = 0

        for gt_bbox in gt_car_bbox_list:
            for dt_bbox in dt_car_bbox_list:
                #print("iou: ", compute_iou_3d(gt_bbox, dt_bbox))
                if compute_iou_3d(gt_bbox, dt_bbox[0:7]) > iou_thred:
                    num_pred_true += 1
                    break
                else:
                    continue
        num_total_pred_true += num_pred_true
        num_total_label_car += num_car

        atk_success_rate = (num_car - num_pred_true) / num_car
        #print("total car: ", num_total_label_car)
        #print("total pred: ", num_total_pred_true)
        # print(atk_success_rate)
        total_atk_success_rate = (num_total_label_car - num_total_pred_true) / num_total_label_car
        pbar.set_postfix({"total_asr": total_atk_success_rate})     
    total_atk_success_rate = (num_total_label_car - num_total_pred_true) / num_total_label_car
    print(f"total_atk_rate: {total_atk_success_rate}")
    # print(f"total_det_rate: {1 - total_atk_success_rate}")
    return None


def parse_arguments():  
    parser = argparse.ArgumentParser(description="计算攻击成功率")  
      
    # 添加参数  
    # parser.add_argument("--label_dir", type=Path, default='./output/Kitti/pv_rcnn/',  help="标签目录的路径")  
    parser.add_argument("--label_dir", type=Path, default='/home/data/GY/final_data/KITTI/water/T1_best_para/label',  help="标签目录的路径")  
    parser.add_argument("--calib_dir", type=Path, default='/home/data/GY/final_data/KITTI/water/T1_best_para/calib', help="标定目录的路径")  
    parser.add_argument("--result_file", type=Path, default='/home/data/GY/final_data/KITTI/water/T1_best_para/pvrcnn_det_result.pkl', help="结果文件的路径")  
    parser.add_argument("--iou_thred", type=float, default=0.7, help="IOU阈值，默认为0.7")  
      
    # 解析参数  
    args = parser.parse_args()  
      
    return args  
  
if __name__ == "__main__":  
    # 解析参数  
    args = parse_arguments()  
      
    # 调用 compute_atk_success_rate 函数并传递解析的参数  
    compute_atk_success_rate(args.label_dir, args.calib_dir, args.result_file, args.iou_thred)
