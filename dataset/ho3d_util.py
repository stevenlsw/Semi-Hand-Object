import numpy as np
import json
import os
import cv2
from PIL import Image


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def load_names(image_path):
    with open(image_path) as f:
        return [line.strip() for line in f]


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


def read_RGB_img(base_dir, seq_name, file_id, split):
    img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')
    img = Image.open(img_filename).convert("RGB")
    return img


def read_gray_img(base_dir, seq_name, file_id):
    img_filename = os.path.join(base_dir, seq_name, file_id + '.png')
    img = Image.open(img_filename)
    return img


def pose_from_RT(R, T):
    pose = np.zeros((4, 4))
    pose[:3, 3] = T
    pose[3, 3] = 1
    R33, _ = cv2.Rodrigues(R)
    pose[:3, :3] = R33
    # change from OpenGL coord to normal coord
    pose[1, :] = -pose[1, :]
    pose[2, :] = -pose[2, :]
    return pose


def projectPoints(xyz, K, rt=None):
    xyz = np.array(xyz)
    K = np.array(K)
    if rt is not None:
        uv = np.matmul(K, np.matmul(rt[:3, :3], xyz.T) + rt[:3, 3].reshape(-1, 1)).T
    else:
        uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def load_objects_HO3D(obj_root):
    import trimesh
    object_names = ['011_banana', '021_bleach_cleanser', '003_cracker_box', '035_power_drill', '025_mug',
                    '006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can', '037_scissors', '004_sugar_box']
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, 'points.xyz')
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = np.array(mesh.vertices)
    return all_models


def filter_test_object(mesh_dict, diameter_dict):
    # filter objects in the evaluation dataset
    mesh_dict_out, diameter_dict_out = {},{}
    for k in mesh_dict.keys():
        if k in ['021_bleach_cleanser','006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can']:
            mesh_dict_out[k] = mesh_dict[k]
            diameter_dict_out[k] = diameter_dict[k]
    return mesh_dict_out, diameter_dict_out


def get_unseen_test_object():
    # unseen objects appear in evaluation set but not in training set
    return ['019_pitcher_base']