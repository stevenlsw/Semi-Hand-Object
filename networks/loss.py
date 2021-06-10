import torch
import torch.nn.functional as F


class Joint2DLoss():
    def __init__(self, lambda_joints2d):
        super(Joint2DLoss,self).__init__()
        self.lambda_joints2d = lambda_joints2d

    def compute_loss(self, preds, gts):
        final_loss = 0
        joint_losses = {}
        if type(preds) == list:
            num_stack = len(preds)
            for i, pred in enumerate(preds):
                joints2d_loss = self.lambda_joints2d * F.mse_loss(pred, gts)
                final_loss += joints2d_loss
                if i == num_stack-1:
                    joint_losses["hm_joints2d_loss"] = joints2d_loss.detach().cpu()
            final_loss /= num_stack
        else:
            joints2d_loss = self.lambda_joints2d * F.mse_loss(preds, gts)
            final_loss = joints2d_loss
            joint_losses["hm_joints2d_loss"] = joints2d_loss.detach().cpu()
        return final_loss, joint_losses


class ManoLoss:
    def __init__(self, lambda_verts3d=None, lambda_joints3d=None,
                 lambda_manopose=None, lambda_manoshape=None,
                 lambda_regulshape=None, lambda_regulpose=None):
        self.lambda_verts3d = lambda_verts3d
        self.lambda_joints3d = lambda_joints3d
        self.lambda_manopose = lambda_manopose
        self.lambda_manoshape = lambda_manoshape
        self.lambda_regulshape = lambda_regulshape
        self.lambda_regulpose = lambda_regulpose

    def compute_loss(self, preds, gts):
        final_loss = 0
        mano_losses = {}
        if type(preds) == list:
            num_stack = len(preds)
            for i, pred in enumerate(preds):
                if self.lambda_verts3d is not None and "verts3d" in gts:
                    mesh3d_loss = self.lambda_verts3d * F.mse_loss(pred["verts3d"], gts["verts3d"])
                    final_loss += mesh3d_loss
                    if i == num_stack - 1:
                        mano_losses["mano_mesh3d_loss"] = mesh3d_loss.detach().cpu()
                if self.lambda_joints3d is not None and "joints3d" in gts:
                    joints3d_loss = self.lambda_joints3d * F.mse_loss(pred["joints3d"], gts["joints3d"])
                    final_loss += joints3d_loss
                    if i == num_stack - 1:
                        mano_losses["mano_joints3d_loss"] = joints3d_loss.detach().cpu()
                if self.lambda_manopose is not None and "mano_pose" in gts:
                    pose_param_loss = self.lambda_manopose * F.mse_loss(pred["mano_pose"], gts["mano_pose"])
                    final_loss += pose_param_loss
                    if i == num_stack - 1:
                        mano_losses["manopose_loss"] = pose_param_loss.detach().cpu()
                if self.lambda_manoshape is not None and "mano_shape" in gts:
                    shape_param_loss = self.lambda_manoshape * F.mse_loss(pred["mano_shape"], gts["mano_shape"])
                    final_loss += shape_param_loss
                    if i == num_stack - 1:
                        mano_losses["manoshape_loss"] = shape_param_loss.detach().cpu()
                if self.lambda_regulshape:
                    shape_regul_loss = self.lambda_regulshape * F.mse_loss(pred["mano_shape"], torch.zeros_like(pred["mano_shape"]))
                    final_loss += shape_regul_loss
                    if i == num_stack - 1:
                        mano_losses["regul_manoshape_loss"] = shape_regul_loss.detach().cpu()
                if self.lambda_regulpose:
                    pose_regul_loss = self.lambda_regulpose * F.mse_loss(pred["mano_pose"][:, 3:], torch.zeros_like(pred["mano_pose"][:, 3:]))
                    final_loss += pose_regul_loss
                    if i == num_stack - 1:
                        mano_losses["regul_manopose_loss"] = pose_regul_loss.detach().cpu()
            final_loss /= num_stack
            mano_losses["mano_total_loss"] = final_loss.detach().cpu()
        else:
            if self.lambda_verts3d is not None and "verts3d" in gts:
                mesh3d_loss = self.lambda_verts3d * F.mse_loss(preds["verts3d"], gts["verts3d"])
                final_loss += mesh3d_loss
                mano_losses["mano_mesh3d_loss"] = mesh3d_loss.detach().cpu()
            if self.lambda_joints3d is not None and "joints3d" in gts:
                joints3d_loss = self.lambda_joints3d * F.mse_loss(preds["joints3d"], gts["joints3d"])
                final_loss += joints3d_loss
                mano_losses["mano_joints3d_loss"] = joints3d_loss.detach().cpu()
            if self.lambda_manopose is not None and "mano_pose" in gts:
                pose_param_loss = self.lambda_manopose * F.mse_loss(preds["mano_pose"], gts["mano_pose"])
                final_loss += pose_param_loss
                mano_losses["manopose_loss"] = pose_param_loss.detach().cpu()
            if self.lambda_manoshape is not None and "mano_shape" in gts:
                shape_param_loss = self.lambda_manoshape * F.mse_loss(preds["mano_shape"], gts["mano_shape"])
                final_loss += shape_param_loss
                mano_losses["manoshape_loss"] = shape_param_loss.detach().cpu()
            if self.lambda_regulshape:
                shape_regul_loss = self.lambda_regulshape * F.mse_loss(preds["mano_shape"], torch.zeros_like(preds["mano_shape"]))
                final_loss += shape_regul_loss
                mano_losses["regul_manoshape_loss"] = shape_regul_loss.detach().cpu()
            if self.lambda_regulpose:
                pose_regul_loss = self.lambda_regulpose * F.mse_loss(preds["mano_pose"][:, 3:], torch.zeros_like(preds["mano_pose"][:, 3:]))
                final_loss += pose_regul_loss
                mano_losses["regul_manopose_loss"] = pose_regul_loss.detach().cpu()
            mano_losses["mano_total_loss"] = final_loss.detach().cpu()
        return final_loss, mano_losses


class ObjectLoss:
    def __init__(self, obj_reg_loss_weight, obj_conf_loss_weight=None, obj_loss_func=None):
        if obj_conf_loss_weight is None and obj_reg_loss_weight is not None:
            obj_conf_loss_weight = obj_reg_loss_weight / 5
        if obj_loss_func is None:
            obj_loss_func = torch.nn.L1Loss()
        self.obj_loss_func = obj_loss_func
        self.obj_conf_loss_weight = obj_conf_loss_weight
        self.obj_reg_loss_weight = obj_reg_loss_weight

    def compute_loss(self, obj_p2d_gt, obj_mask, obj_pred, obj_lossmask=None):
        obj_losses = {}
        # get predictions for output
        reg_px = obj_pred[0]
        reg_py = obj_pred[1]
        reg_conf = obj_pred[2]
        mask_front = obj_mask.repeat(21, 1, 1, 1).permute(1, 2, 3, 0).contiguous()
        reg_py = reg_py * mask_front
        reg_px = reg_px * mask_front
        reg_label = obj_p2d_gt.repeat(32, 32, 1, 1, 1).permute(2, 0, 1, 3, 4).contiguous()
        reg_label_x = reg_label[:, :, :, :, 0]
        reg_label_y = reg_label[:, :, :, :, 1]
        reg_label_x = reg_label_x * mask_front
        reg_label_y = reg_label_y * mask_front

        # confidence regression result
        bias = torch.sqrt((reg_py - reg_label_y) ** 2 + (reg_px - reg_label_x) ** 2)
        conf_target = torch.exp(-1 * bias) * mask_front
        conf_target = conf_target.detach()

        if obj_lossmask is None:
            reg_loss = self.obj_loss_func(reg_px, reg_label_x) + self.obj_loss_func(reg_py, reg_label_y)
            conf_loss = self.obj_loss_func(reg_conf, conf_target)
        else:
            obj_lossmask = obj_lossmask.view(-1, 1, 1, 1)
            reg_px = reg_px * obj_lossmask
            reg_py = reg_py * obj_lossmask
            reg_label_x = reg_label_x * obj_lossmask
            reg_label_y = reg_label_y * obj_lossmask
            reg_conf = reg_conf * obj_lossmask
            conf_target = conf_target * obj_lossmask
            reg_loss = self.obj_loss_func(reg_px, reg_label_x) + self.obj_loss_func(reg_py, reg_label_y)
            conf_loss = self.obj_loss_func(reg_conf, conf_target)
            if obj_lossmask.sum() != 0:
                reg_loss *= reg_px.shape[0] / obj_lossmask.sum()
                conf_loss *= reg_px.shape[0] / obj_lossmask.sum()

        reg_loss = self.obj_reg_loss_weight * reg_loss
        conf_loss = self.obj_conf_loss_weight * conf_loss
        obj_losses["obj_reg_loss"] = reg_loss.detach().cpu()
        obj_losses["obj_conf_loss"] = conf_loss.detach().cpu()
        final_loss = reg_loss + conf_loss
        return final_loss, obj_losses