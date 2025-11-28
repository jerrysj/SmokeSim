import os
from pathlib import Path
import generate_adv_dataset
import run_model_and_save
import compute_atk_metric


# kitti_dir = "/home/data/kitti/detection/training/" 
# bin_dir   = os.path.join(kitti_dir, "velodyne")
# label_dir = os.path.join(kitti_dir, "label_2")
# calib_dir = os.path.join(kitti_dir, "calib")
# fog_dir   = "./fog_yanwu/"
# out_dir = '/home/data/GY/final_data/KITTI/smoke/TEST_heng_0.002_zong_0.25/'
# # out_dir = "./output/Kitti/"
# 
# # generate_adv_dataset.mix_fog_to_kitti_dataset(bin_dir, 
# #                                               label_dir, 
# #                                               calib_dir, 
# #                                               fog_dir, 
# #                                               out_dir)

kitti_dir = "/home/data/kitti/detection/training/" 
bin_dir   = os.path.join(kitti_dir, "velodyne")
label_dir = os.path.join(kitti_dir, "label_2")
calib_dir = os.path.join(kitti_dir, "calib")
# fog_dir   = "./fog_yanwu_unity/"
# fog_dir   = "./fog_for_figure/"
# fog_dir   = "./fog_yanwu/"
fog_dir = "./fog_random_noise/"
out_dir = '/home/data/GY/final_data/KITTI/random_noise/TEST_heng_0.002_zong_0.25'

generate_adv_dataset.mix_fog_to_kitti_dataset(
    Path(bin_dir), 
    Path(label_dir), 
    Path(calib_dir), 
    Path(fog_dir), 
    Path(out_dir)
)

# exit(1)

cfgs_dir  = './third_party/OpenPCDet/tools/cfgs/kitti_models/'
cfgs_file = os.path.join(cfgs_dir, 'PartA2.yaml')
# cfgs_file = os.path.join(cfgs_dir, 'PartA2_free.yaml')
# cfgs_file = os.path.join(cfgs_dir, 'pv_rcnn.yaml')
# cfgs_file = os.path.join(cfgs_dir, 'pointpillar.yaml')

# data_dir  = '/home/data/kitti/detection/training/velodyne/'
data_dir  = os.path.join(out_dir, 'adv_data')

ckpt_dir  = './third_party/PretrainModel/kitti'
ckpt_file = os.path.join(ckpt_dir, 'PartA2_7940.pth')
# ckpt_file = os.path.join(ckpt_dir, 'PartA2_free_7872.pth')
# ckpt_file = os.path.join(ckpt_dir, 'pv_rcnn_8369.pth')
# ckpt_file = os.path.join(ckpt_dir, 'pointpillar_7728.pth')

save_name = 'partA2_anchor'
# save_name = 'partA2_free'
# save_name = 'pvrcnn'
# save_name = 'pointpillar'
run_model_and_save.run_kitti_model_and_save(
    Path(cfgs_file),
    Path(data_dir),
    Path(ckpt_file),
    '.bin',
    Path(out_dir),
    save_name
)

#label_dir   = os.path.join(out_dir, "label")
label_dir   = os.path.join('/home/data/GY/final_data/KITTI/nomal_pre_det/', 'Part_A2_Anchor')
# label_dir   = os.path.join('/home/data/GY/final_data/KITTI/nomal_pre_det/', 'Part_A2_Free')
# label_dir   = os.path.join('/home/data/GY/final_data/KITTI/nomal_pre_det/', 'PV_RCNN')
# label_dir   = os.path.join('/home/data/GY/final_data/KITTI/nomal_pre_det/', 'PointPillar')
calib_dir   = os.path.join(out_dir, "calib")
# label_dir   = os.path.join(kitti_dir, "label_2")
# calib_dir   = os.path.join(kitti_dir, "calib")
result_file = os.path.join(out_dir, f"{save_name}_det_result.pkl")
iou_thred   = 0.7
compute_atk_metric.compute_atk_success_rate(
    Path(label_dir),
    Path(calib_dir),
    Path(result_file),
    iou_thred,
    save_name,
)
