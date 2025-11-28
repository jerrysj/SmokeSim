import os, shutil
from os import PathLike
from nuscenes.nuscenes import NuScenes
import numpy as np
import pickle as pkl

from utils.nuscenes import nusc_utils
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

from utils.kitti import kitti_utils

def mix_fog_to_kitti_dataset(
    bin_dir:   PathLike,
    label_dir: PathLike,
    calib_dir: PathLike,
    fog_dir:   PathLike,
    out_dir:   PathLike,
    use_ri: bool =True,
    with_intensity: bool = False,
):
    bin_list   = os.listdir(bin_dir)
    fog_list   = os.listdir(fog_dir)
    label_list = os.listdir(label_dir)
    calib_list = os.listdir(calib_dir)

    bin_list.sort()
    fog_list.sort()
    label_list.sort()
    calib_list.sort()

    if not os.path.exists(out_dir):
        print("create output dir: ", out_dir)
        os.makedirs(os.path.join(out_dir, 'adv_data'))
        os.makedirs(os.path.join(out_dir, 'calib'))
        os.makedirs(os.path.join(out_dir, 'label'))
    else:
        print("output dir is exists")
        cur_dir = os.getcwd()
        os.chdir(out_dir)
        if not os.path.exists('adv_data'): 
            os.mkdir('adv_data')
        if not os.path.exists('adv_data'): 
            os.mkdir('calib')
        if not os.path.exists('adv_data'): 
            os.mkdir('label')
        os.chdir(cur_dir)

    # for i in range(len(bin_list)):
    for i in range(300):
        bin_prefix   = os.path.splitext(bin_list[i])[0]
        label_prefix = os.path.splitext(label_list[i])[0]
        calib_prefix = os.path.splitext(calib_list[i])[0]
        assert bin_prefix == label_prefix and bin_prefix == calib_prefix, \
               f"file prefix should be equal, but {bin_prefix} != {label_prefix} != {calib_prefix}"
        
        bin_file   = os.path.join(bin_dir, bin_list[i])
        calib_file = os.path.join(calib_dir, calib_list[i])
        label_file = os.path.join(label_dir, label_list[i])

        kitti_utils.init(calib_file)
        bboxs, btypes = kitti_utils.kitti_read_bbox(label_file)

        points_with_i   = kitti_utils.kitti_read_points_clouds(bin_file)
        points = points_with_i[:, :3]
        car_bbox = [bboxs[i] for i in range(len(btypes)) if btypes[i] in ['Car']]

        if car_bbox == []:
            print(f"adv {bin_prefix} does not have car object, skip...")
            continue

        for fog_idx, fog_file in enumerate(fog_list):
            fog_points   = np.fromfile(os.path.join(fog_dir, fog_file), sep=' ', dtype=np.float32).reshape(-1, 3)
            fog_centroid = np.mean(fog_points, axis=0)
            
            if use_ri:
                if with_intensity:
                    _, points = kitti_utils.mix_fog_to_car(points_with_i, car_bbox, fog_points, fog_centroid, return_pc=True, with_intensity=True)
                    new_points = points
                else:
                    range_image  = kitti_utils.mix_fog_to_car(points, car_bbox, fog_points, fog_centroid)
                    new_points   = kitti_utils.ri2points(range_image)
            else:
                _, points = kitti_utils.mix_fog_to_car(points, car_bbox, fog_points, fog_centroid, return_pc=True, with_intensity=False)
                new_points = points
            # ================ TEST ==================
            # new_points = points
            # ================ TEST ==================
            new_points   = np.hstack((new_points, np.zeros((new_points.shape[0], 1))))

            new_bin_file   = os.path.join(out_dir, f"adv_data/{bin_prefix}_{fog_idx}.bin")
            new_calib_file = os.path.join(out_dir, f"calib/{calib_prefix}_{fog_idx}.txt")
            new_label_file = os.path.join(out_dir, f"label/{label_prefix}_{fog_idx}.txt")

            print(f"adv data file {bin_prefix}_{fog_idx} will be created...")
            new_points.astype(np.float32).tofile(new_bin_file)
            shutil.copy(label_file, new_label_file)
            shutil.copy(calib_file, new_calib_file)

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    See https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    a = 2.0 * (q[0] * q[3] + q[1] * q[2])
    b = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)

    return np.arctan2(a, b)

def mix_fog_to_nusc_dataset(version:  str,
                            dataroot: PathLike,
                            fog_dir:  PathLike,
                            out_dir:  PathLike,
                            verbose:  bool = True,
                            use_ri=True,
                            with_intensity=False,
                            ):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)

    fog_list = os.listdir(fog_dir)
    fog_list.sort()

    if not os.path.exists(out_dir):
        print("create output dir: ", out_dir)
        os.makedirs(os.path.join(out_dir, 'adv_data'))
        os.makedirs(os.path.join(out_dir, 'label'))
    else:
        print("output dir is exists")
        cur_dir = os.getcwd()
        os.chdir(out_dir)
        if not os.path.exists('adv_data'): 
            os.mkdir('adv_data')
        if not os.path.exists('adv_data'): 
            os.mkdir('label')
        os.chdir(cur_dir)

    # for i in range(len(nusc.sample)):
    # for i in range(3000):
    for i in range(300):
        sample_name = "%06d" % i
        sample = nusc.sample[i]
        lidar_sample_data = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        lidar_file        = os.path.join(nusc.dataroot, lidar_sample_data["filename"])
        points_with_i   = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)
        points_l          = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)[:, :3]
        ego_pose          = nusc.get('ego_pose', lidar_sample_data['ego_pose_token'])
        ego_to_global     = nusc_utils.to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, ego_pose['translation'], False)
        lidar_sensor      = nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
        lidar_to_ego      = nusc_utils.to_matrix4x4_2(Quaternion(lidar_sensor['rotation']).rotation_matrix, lidar_sensor['translation'], False)
        lidar_to_global   = ego_to_global @ lidar_to_ego

        # ================= TEMP ========================
        # new_bin_file = os.path.join(out_dir, 'adv_data', f"{sample_name}.npy")
        # points_l          = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)[:, :3]
        # points_l  = np.hstack((points_l, np.zeros((points_l.shape[0], 2))))  
        # print(f"adv data file {sample_name} will be created...")
        # np.save(new_bin_file, points_l)
        # # 这里开始保存标签
        # new_car_bbox_list = []
        # temp_car_bbox_list = nusc.get_sample_data(lidar_sample_data['token'])[1]
        # for bbox in temp_car_bbox_list:
        #     if bbox.name != 'vehicle.car':
        #         continue
        #     x, y, z = bbox.center
        #     w, l, h = bbox.wlh
        #     ry = quaternion_yaw(bbox.orientation)
        #     #ry = -ry + np.pi / 2
        #     new_car_bbox_list.append(
        #         [x, y, z, l, w, h, ry]
        #     )
        # new_label_file = os.path.join(out_dir, 'label', f"{sample_name}.pkl")
        # with open(new_label_file, 'wb') as f:
        #         pkl.dump(new_car_bbox_list, f)
        # continue
        # ================= TEMP ========================
        # car_bbox_list = []
        # for annotation_token in sample['anns']:
        #     instance = nusc.get('sample_annotation', annotation_token)
        #     if instance['category_name'] == 'vehicle.car':
        #         box = Box(instance['translation'], instance['size'], Quaternion(instance['rotation']))
        #         car_bbox_list.append(box)
        
        # if car_bbox_list == []:
        #     print(f"file {os.path.basename(lidar_file)} does not have car object, skip...")
        #     continue

        car_bbox_list = []
        new_car_bbox_list = []
        temp_car_bbox_list = nusc.get_sample_data(lidar_sample_data['token'])[1]
        for bbox in temp_car_bbox_list:
            if bbox.name != 'vehicle.car':
                continue
            x, y, z = bbox.center
            w, l, h = bbox.wlh
            car_bbox_list.append(bbox)
            ry = quaternion_yaw(bbox.orientation)
            #ry = -ry + np.pi / 2
            new_car_bbox_list.append(
                [x, y, z, l, w, h, ry]
            )

        if new_car_bbox_list == []:
            print(f"file {os.path.basename(lidar_file)} does not have car object, skip...")
            continue

        for fog_idx, fog_file in enumerate(fog_list):
            fog_points   = np.fromfile(os.path.join(fog_dir, fog_file), sep=' ', dtype=np.float32).reshape(-1, 3)
            fog_centroid = np.mean(fog_points, axis=0)
            
            # range_image = nusc_utils.mix_fog_to_car(points_l, car_bbox_list, fog_points, fog_centroid, lidar_to_global)
            if use_ri:
                if with_intensity:
                    _, points = nusc_utils.mix_fog_to_car(points_with_i, car_bbox_list, fog_points, fog_centroid, lidar_to_global, return_pc=True, with_intensity=True)
                    new_points = points
                else:
                    range_image  = nusc_utils.mix_fog_to_car(points_l, car_bbox_list, fog_points, fog_centroid, lidar_to_global)
                    new_points, _   = nusc_utils.ri2points(range_image)
            else:
                _, points = nusc_utils.mix_fog_to_car(points_l, car_bbox_list, fog_points, fog_centroid, lidar_to_global, return_pc=True, with_intensity=False)
                new_points   = points

            new_points  = np.hstack((new_points, np.zeros((new_points.shape[0], 2))))  

            new_bin_file   = os.path.join(out_dir, f"adv_data/{sample_name}_{fog_idx}.npy")
            new_label_file = os.path.join(out_dir, f"label/{sample_name}_{fog_idx}.pkl")

            print(f"adv data file {sample_name}_{fog_idx} will be created...")
            #new_points.astype(np.float32).tofile(new_bin_file)
            np.save(new_bin_file, new_points)

            with open(new_label_file, 'wb') as f:
                pkl.dump(new_car_bbox_list, f)
