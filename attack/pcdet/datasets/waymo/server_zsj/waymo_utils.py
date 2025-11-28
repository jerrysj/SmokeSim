# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import pickle
import numpy as np
from ...utils import common_utils
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
import random

try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']


def generate_labels(frame, pose):
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    # NOTE: 这个是自车坐标系下的
    #       Ref: https://github.com/waymo-research/waymo-open-dataset/issues/736
    laser_labels = frame.laser_labels
    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)
        speeds.append([laser_labels[i].metadata.speed_x, laser_labels[i].metadata.speed_y])
        accelerations.append([laser_labels[i].metadata.accel_x, laser_labels[i].metadata.accel_y])

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)
    annotations['speed_global'] = np.array(speeds)
    annotations['accel_global'] = np.array(accelerations)

    annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        global_speed = np.pad(annotations['speed_global'], ((0, 0), (0, 1)), mode='constant', constant_values=0)  # (N, 3)
        speed = np.dot(global_speed, np.linalg.inv(pose[:3, :3].T))
        speed = speed[:, :2]
        
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis], speed],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 9))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3] # TOP: H = 64, W = 2650
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def save_lidar_points(frame, cur_save_path, use_two_returns=True):
    ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(ret_outputs) == 4:
        range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
    else:
        assert len(ret_outputs) == 3
        range_images, camera_projections, range_image_top_pose = ret_outputs

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1) if use_two_returns else (0,)
    )

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)

    np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)
    return num_points_of_each_lidar

def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True, use_two_returns=True, update_info_only=False):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    # NOTE: 从 `.tfrecord`
    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    sequence_infos = []
    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        sequence_infos_old = None
        if not update_info_only:
            print('Skip sequence since it has been processed before: %s' % pkl_file)
            return sequence_infos
        else:
            sequence_infos_old = sequence_infos
            sequence_infos = []

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        # NOTE: 一帧的数据
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame, pose=pose)
            info['annos'] = annotations

        if update_info_only and sequence_infos_old is not None:
            assert info['frame_id'] == sequence_infos_old[cnt]['frame_id']
            num_points_of_each_lidar = sequence_infos_old[cnt]['num_points_of_each_lidar']
        else:
            num_points_of_each_lidar = save_lidar_points(
                frame, cur_save_dir / ('%04d.npy' % cnt), use_two_returns=use_two_returns
            )
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos


fog_file_path = '~/../project/zsj/OpenPCDet/data/smoke_aaai2025/smoke_5.txt'
def read_fog_pc(file=fog_file_path):
    fog_pc = np.loadtxt(os.path.expanduser(file), dtype=np.float32) 
    fog_pc = fog_pc - np.mean(fog_pc, axis=0)
    fog_pc[:, 2] -= np.min(fog_pc[:, 2])
    return  fog_pc

from .debug_utils.visual import show_scene_pc, add_bboxs_to_fig

from .ops import vehicle_to_local_coor, get_bbox_mask, transform_fog_to_scene_pc

def rotate_points_along_z(points, angle):
    """
    将点绕z轴旋转给定的角度
    """
    cosa = np.math.cos(angle)
    sina = np.math.sin(angle)
    rot_matrix = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])
    return np.dot(points, rot_matrix.T)

def get_corners_from_label_bbox(bbox):
    """
    generate corners3d representation for this object

                 z
          5 -----|-------- 4   
         /|      |        /|   
        1 -------------- 0 .   
        | |   box|_______|_|__ y
        . 6 -----/-------| 7   
        |/      /        |/    
        2 -----/---------3     
              x                
                               
    """
    dx, dy, dz = bbox[3:6]
    #             0        1        2        3        4        5        6        7
    x_corners = [+dx / 2, +dx / 2, +dx / 2, +dx / 2, -dx / 2, -dx / 2, -dx / 2, -dx / 2]
    y_corners = [+dy / 2, -dy / 2, -dy / 2, +dy / 2, +dy / 2, -dy / 2, -dy / 2, +dy / 2]
    z_corners = [+dz / 2, +dz / 2, -dz / 2, -dz / 2, +dz / 2, +dz / 2, -dz / 2, -dz / 2]

    corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    corners = rotate_points_along_z(corners.T, bbox[6])
    corners = corners + bbox[0:3]
    return corners

def extract_points_by_bbox(points_l: np.ndarray, bbox) -> np.ndarray:
    """
    Args:
        points_l: [N, 3], [x, y, z]
        car_box: [1, 7], [x, y, z, dx, dy, dz, heading]
    """
    points_b = lidar_to_box(points_l, bbox)

    x, y, z    = points_b[:, 0], points_b[:, 1], points_b[:, 2]
    dx, dy, dz = bbox[3], bbox[4], bbox[5]
    
    is_within_box = (abs(x) < dx/2) & (abs(y) < dy / 2) & (abs(z) < dz/2)
    car_points_l = points_l[is_within_box]
    car_points_b = points_b[is_within_box]
    return car_points_l, car_points_b

def lidar_to_box(points_l, bbox):
    trans = np.array([bbox[0:3]])

    points_b_mat = rotate_points_along_z(points_l - trans, -bbox[6])
    points_b = points_b_mat # [N, 3]
    return points_b

def downsample_fog_pc(fog_pc, num_points):
    if fog_pc.shape[0] > num_points:
        indices = np.random.choice(fog_pc.shape[0], num_points, replace=False)
        fog_pc = fog_pc[indices]
    return fog_pc

# TODO: 贴车身 + 下采样
def mix_points_and_fog_pc(points, fog_pc, laser_labels, offset = 0.):
    """
                 z
          5 -----|-------- 4   
         /|      |        /|   
        1 -------------- 0 .   
        | |   box|_______|_|__ y
        . 6 -----/-------| 7   
        |/      /        |/    
        2 -----/---------3     
              x                
                               
    face1: [0, 1, 2, 3]
    face2: [4, 5, 6, 7]
    face3: [0, 4, 7, 3]
    face4: [1, 5, 6, 2]
    """
    # NOTE: 现在 points 和 annotations 都是在自车坐标系下
    car_bboxs = []
    for laser_label in laser_labels:
        if laser_label.type == 1 or (laser_label.type == 2 and random.random() < 0.3):
            x = laser_label.box.center_x
            y = laser_label.box.center_y
            z = laser_label.box.center_z
            w = laser_label.box.width
            l = laser_label.box.length
            h = laser_label.box.height
            heading = laser_label.box.heading
            car_bboxs.append([x, y, z, l, w, h, heading])
        else:
            continue

    # CHECK: looks good!
    # fig = show_scene_pc(points[:-3000])
    # fig = add_bboxs_to_fig(fig, car_bboxs)
    # fig.show()

    # NOTE: 对 fog_pc 做下采样
    # fog_pc = downsample_fog_pc(fog_pc, 1024)
    fog_pc = downsample_fog_pc(fog_pc, 1500)
    
    # NOTE: 开始加烟雾
    pc_xyz, other_feat = points[:, :3], points[:, 3:]  # 分离位置和特征
    smoke_xyz = np.zeros((0, 3), dtype=np.float32)
    for car_bbox in car_bboxs:
        # NOTE: BEV 视图上，车中心点(原点)和 bbox 中心点连线，把烟雾放在这条线上距离车 offset 米处的地方
        corners = get_corners_from_label_bbox(car_bbox)
        p0, p1, p2, p3, p4, p5, p6, p7 = corners

        # NOTE: 判断哪两个是边长较长的那两个侧面
        len1 = np.linalg.norm(p2 - p3) # face1, face2
        len2 = np.linalg.norm(p2 - p6) # face3, face4

        if len1 > len2:
            # face1, face2
            side_1, side_2 = corners[:4], corners[4:]
        else:
            # face3, face4
            side_1 = [corners[i] for i in [0, 4, 7, 3]]
            side_2 = [corners[i] for i in [1, 5, 6, 2]]

        side_1, side_2 = np.array(side_1), np.array(side_2)

        # NOTE: 判断哪个面是被激光雷达扫描到的那一面
        _, std_xyz = extract_points_by_bbox(pc_xyz, car_bbox)

        if len1 > len2:
            # NOTE: 标记每个点是更靠近正方向的一面还是负方向的一面
            # side1 = face1(x 轴正方向), side2 = face2(x 轴负方向)，
            is_close_pos_side = std_xyz[:, 0] > 0
            num_close_pos_side = np.count_nonzero(is_close_pos_side)
        else:
            # side1 = face3(y 轴正方向), side2 = face4(y 轴负方向)
            is_close_pos_side = std_xyz[:, 1] > 0
            num_close_pos_side = np.count_nonzero(is_close_pos_side)

        if num_close_pos_side > std_xyz.shape[0] / 2:
            # face1(x 轴正方向) or face3(y 轴正方向)
            side = side_1
        else:
            # face2(x 轴负方向) or face4(y 轴负方向)
            side = side_2

        fog_set_place = (side[2] + side[3]) / 2
        fog_pc_ = transform_fog_to_scene_pc([fog_pc], fog_set_place)[0]
        smoke_xyz = np.concatenate([smoke_xyz, fog_pc_[:, :3]], axis=0)  # 更新 xyz
    # NOTE: DEBUG 使用
    # np.savetxt("./a.txt", np.concatenate([smoke_xyz, pc_xyz], axis=0), fmt="%.6f")
    # exit(0)
    return np.concatenate([smoke_xyz, pc_xyz], axis=0)

    #     x, y = car_bbox[0], car_bbox[1]
    #     bbox_bev_center = np.array([x, y, 0])
    #     distance = np.sqrt(x**2 + y**2)
    #     fog_set_place = bbox_bev_center * (1 - offset / distance)

    #     heading = car_bbox[6]
    #     fog_pc_ = transform_fog_to_scene_pc(fog_pc, fog_set_place)
    #     points = np.concatenate([points, fog_pc_], axis=0)

    # return points

# def mix_points_and_fog_pc(points, fog_pc, laser_labels, offset = 6.):
#     # NOTE: 现在 points 和 annotations 都是在自车坐标系下
#     car_bboxs = []
#     for laser_label in laser_labels:
#         if laser_label.type == 1 or (laser_label.type == 2 and random.random() < 0.3):
#             x = laser_label.box.center_x
#             y = laser_label.box.center_y
#             z = laser_label.box.center_z
#             w = laser_label.box.width
#             l = laser_label.box.length
#             h = laser_label.box.height
#             heading = laser_label.box.heading
#             car_bboxs.append([x, y, z, l, w, h, heading])
#         else:
#             continue
# 
#     # CHECK: looks good!
#     # fig = show_scene_pc(points[:-3000])
#     # fig = add_bboxs_to_fig(fig, car_bboxs)
#     # fig.show()
#     
#     # NOTE: 开始加烟雾
#     for car_bbox in car_bboxs:
#         # NOTE: BEV 视图上，车中心点(原点)和 bbox 中心点连线，把烟雾放在这条线上距离车 offset 米处的地方
#         x, y = car_bbox[0], car_bbox[1]
#         bbox_bev_center = np.array([x, y, 0])
#         distance = np.sqrt(x**2 + y**2)
#         fog_set_place = bbox_bev_center * (1 - offset / distance)
# 
#         heading = car_bbox[6]
#         fog_pc_ = transform_fog_to_scene_pc(fog_pc, fog_set_place)
#         points = np.concatenate([points, fog_pc_], axis=0)
# 
#     return points

def save_lidar_points_with_fog(frame, cur_save_path, use_two_returns=True):
    ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(ret_outputs) == 4:
        range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
    else:
        assert len(ret_outputs) == 3
        range_images, camera_projections, range_image_top_pose = ret_outputs

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1) if use_two_returns else (0,)
    )

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    fog_pc = read_fog_pc(file=fog_file_path)
    
    # NOTE: 将 fog 点云放到 points 中某个位置得到混合点云, 同时填充相应的数据
    #       强度和伸长率全置为零就好, NLZ_flag 也一样置零
    mix_points = mix_points_and_fog_pc(points_all, fog_pc, frame.laser_labels)
    mix_points = mix_points.astype(np.float32)
    # np.savetxt('before_proj.txt', mix_points, fmt='%.6f')
    # points_in_NLZ_flag = np.concatenate([points_in_NLZ_flag, np.zeros((mix_points.shape[0] - points_all.shape[0], 1), dtype=np.float32)], axis=0)
    # points_intensity   = np.concatenate([points_intensity,   np.zeros((mix_points.shape[0] - points_all.shape[0], 1), dtype=np.float32)], axis=0)
    # points_elongation  = np.concatenate([points_elongation,  np.zeros((mix_points.shape[0] - points_all.shape[0], 1), dtype=np.float32)], axis=0)
    # point_features     = np.concatenate([points_in_NLZ_flag, points_intensity, points_elongation], axis=-1)# [N, 3]

    # NOTE: 投影成 Range Image 排除重复点
    #       使用 range_image_utils 里面的函数就行
    #       投影到车顶雷达的坐标系，然后反投回自车坐标系下的点云
    top_c = frame.context.laser_calibrations[-1]
    extrinsic = np.reshape(np.array(top_c.extrinsic.transform), [4, 4])
    beam_inclinations = tf.constant(top_c.beam_inclinations)
    range_image_tensor = range_image_utils.build_range_image_from_point_cloud(
        points_vehicle_frame = tf.expand_dims(tf.convert_to_tensor(mix_points), axis=0),
        num_points = tf.convert_to_tensor([len(mix_points)]),
        extrinsic = tf.expand_dims(extrinsic, axis=0),  
        inclination = tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
        range_image_size = [64, 2650],
        # point_features = tf.expand_dims(tf.convert_to_tensor(point_features), axis=0),
    )[0]
    range_image_mask = range_image_tensor[0] > 0
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        range_image = range_image_tensor,
        extrinsic = tf.expand_dims(extrinsic, axis=0),  
        inclination = tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
    )
    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    points_tensor = tf.gather_nd(range_image_cartesian, tf.where(range_image_mask))
    # np.savetxt('after_proj.txt', points_tensor.numpy(), fmt='%.6f')
    # exit(1)

    # CHECK: looks good!
    # fig = show_scene_pc(points_tensor[:-3000])
    # fig.show()

    # FIXME: 这里可能会有问题，因为现在只提供了一颗雷达的数据
    num_points_of_each_lidar = [points_tensor.shape[0]]
    max_length = points_tensor.shape[0]
    points_intensity = np.zeros((max_length, 1), dtype=np.float32)
    points_elongation = np.zeros((max_length, 1), dtype=np.float32)
    points_in_NLZ_flag = np.zeros((max_length, 1), dtype=np.float32)
    save_points = np.concatenate([
        points_tensor.numpy(), points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)

    np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)
    return num_points_of_each_lidar

def process_single_sequence_with_fog(sequence_file, save_path, sampled_interval, has_label=True, use_two_returns=True, update_info_only=False):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    # NOTE: 从 `.tfrecord`
    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    sequence_infos = []
    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        sequence_infos_old = None
        if not update_info_only:
            print('Skip sequence since it has been processed before: %s' % pkl_file)
            return sequence_infos
        else:
            sequence_infos_old = sequence_infos
            sequence_infos = []

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        # NOTE: 一帧的数据
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame, pose=pose)
            info['annos'] = annotations
        else:
            print(f"未预料的情况，请检查，文件: {__file__}, Line: {inspect.currentframe().f_lineno}")
            assert False

        if update_info_only and sequence_infos_old is not None:
            assert info['frame_id'] == sequence_infos_old[cnt]['frame_id']
            num_points_of_each_lidar = sequence_infos_old[cnt]['num_points_of_each_lidar']
        else:
            num_points_of_each_lidar = save_lidar_points_with_fog(
                frame, cur_save_dir / ('%04d.npy' % cnt), use_two_returns=use_two_returns
            )
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos


