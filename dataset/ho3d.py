import os
from torchvision.transforms import functional
from torch.utils import data
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

from dataset import ho3d_util
from dataset import dataset_util


class HO3D(data.Dataset):
    def __init__(self, dataset_root, obj_model_root, train_label_root="./ho3d-process",
                 mode="evaluation", inp_res=512,
                 max_rot=np.pi, scale_jittering=0.2, center_jittering=0.1,
                 hue=0.15, saturation=0.5, contrast=0.5, brightness=0.5, blur_radius=0.5):
        # Dataset attributes
        self.root = dataset_root
        self.mode = mode
        self.inp_res = inp_res
        self.joint_root_id = 0
        self.jointsMapManoToSimple = [0, 13, 14, 15, 16,
                                      1, 2, 3, 17,
                                      4, 5, 6, 18,
                                      10, 11, 12, 19,
                                      7, 8, 9, 20]
        self.jointsMapSimpleToMano = np.argsort(self.jointsMapManoToSimple)
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        # object informations
        self.obj_mesh = ho3d_util.load_objects_HO3D(obj_model_root)
        self.obj_bbox3d = dataset_util.get_bbox21_3d_from_dict(self.obj_mesh)
        self.obj_diameters = dataset_util.get_diameter(self.obj_mesh)

        if self.mode == "train":
            self.hue = hue
            self.contrast = contrast
            self.brightness = brightness
            self.saturation = saturation
            self.blur_radius = blur_radius
            self.scale_jittering = scale_jittering
            self.center_jittering = center_jittering
            self.max_rot = max_rot

            self.train_seg_root = os.path.join(train_label_root, "train_segLabel")

            self.mano_params = []
            self.joints_uv = []
            self.obj_p2ds = []
            self.K = []
            # training list
            self.set_list = ho3d_util.load_names(os.path.join(train_label_root, "train.txt"))
            # camera matrix
            K_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_K.json'))
            # hand joints
            joints_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_joint.json'))
            # mano params
            mano_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_mano.json'))
            # obj landmarks
            obj_p2d_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_obj.json'))
            for i in range(len(self.set_list)):
                K = np.array(K_list[i], dtype=np.float32)
                self.K.append(K)
                self.joints_uv.append(ho3d_util.projectPoints(np.array(joints_list[i], dtype=np.float32), K))
                self.mano_params.append(np.array(mano_list[i], dtype=np.float32))
                self.obj_p2ds.append(np.array(obj_p2d_list[i], dtype=np.float32))
        else:
            self.set_list = ho3d_util.load_names(os.path.join(self.root, "evaluation.txt"))

    def data_aug(self, img, mano_param, joints_uv, K, gray, p2d):
        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)

        # Randomly jitter center
        center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
        center = center + center_offsets

        # Scale jittering
        scale_jittering = self.scale_jittering * np.random.randn() + 1
        scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
        scale = scale * scale_jittering

        rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(center, scale,
                                                                              [self.inp_res, self.inp_res], rot=rot,
                                                                              K=K)
        # Change mano from openGL coordinates to normal coordinates
        mano_param[:3] = dataset_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=self.coord_change_mat)

        joints_uv = dataset_util.transform_coords(joints_uv, affinetrans)  # hand landmark trans
        K = post_rot_trans.dot(K)

        p2d = dataset_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        # Img blurring and color jitter
        blur_radius = random.random() * self.blur_radius
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = dataset_util.color_jitter(img, brightness=self.brightness,
                                        saturation=self.saturation, hue=self.hue, contrast=self.contrast)

        # Generate object mask: gray segLabel transform and crop
        gray = dataset_util.transform_img(gray, affinetrans, [self.inp_res, self.inp_res])
        gray = gray.crop((0, 0, self.inp_res, self.inp_res))
        gray = dataset_util.get_mask_ROI(gray, bbox_obj)
        # Generate object mask
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        obj_mask = torch.from_numpy(obj_mask)

        return img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj

    def data_crop(self, img, K, bbox_hand, p2d):
        crop_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.1)
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_util.get_affine_transform(center, scale, [self.inp_res, self.inp_res])
        bbox_hand = dataset_util.transform_coords(bbox_hand.reshape(2, 2), affinetrans).flatten()
        bbox_obj = dataset_util.transform_coords(bbox_obj.reshape(2, 2), affinetrans).flatten()
        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        K = affinetrans.dot(K)
        return img, K, bbox_hand, bbox_obj

    def __len__(self):
        return len(self.set_list)

    def __getitem__(self, idx):
        sample = {}
        seqName, id = self.set_list[idx].split("/")
        img = ho3d_util.read_RGB_img(self.root, seqName, id, self.mode)
        if self.mode == 'train':
            K = self.K[idx]
            # hand information
            joints_uv = self.joints_uv[idx]
            mano_param = self.mano_params[idx]
            # object information
            gray = ho3d_util.read_gray_img(self.train_seg_root, seqName, id)
            p2d = self.obj_p2ds[idx]
            # data augmentation
            img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj = self.data_aug(img, mano_param, joints_uv, K, gray, p2d)
            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["mano_param"] = mano_param
            sample["cam_intr"] = K
            sample["joints2d"] = joints_uv
            sample["obj_p2d"] = p2d
            sample["obj_mask"] = obj_mask
        else:
            annotations = np.load(os.path.join(os.path.join(self.root, self.mode), seqName, 'meta', id + '.pkl'),
                                  allow_pickle=True)
            K = np.array(annotations['camMat'], dtype=np.float32)
            # object
            sample["obj_cls"] = annotations['objName']
            sample["obj_bbox3d"] = self.obj_bbox3d[sample["obj_cls"]]
            sample["obj_diameter"] = self.obj_diameters[sample["obj_cls"]]
            obj_pose = ho3d_util.pose_from_RT(annotations['objRot'].reshape((3,)), annotations['objTrans'])
            p2d = ho3d_util.projectPoints(sample["obj_bbox3d"], K, rt=obj_pose)
            sample["obj_pose"] = obj_pose
            # hand 
            bbox_hand = np.array(annotations['handBoundingBox'], dtype=np.float32)
            root_joint = np.array(annotations['handJoints3D'], dtype=np.float32)
            root_joint = root_joint.dot(self.coord_change_mat.T)
            sample["root_joint"] = root_joint
            img, K, bbox_hand, bbox_obj = self.data_crop(img, K, bbox_hand, p2d)
            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["cam_intr"] = K
            
        return sample