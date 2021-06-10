import os
import time
import torch

from utils.utils import progress_bar as bar, AverageMeters, dump
from dataset.ho3d_util import filter_test_object, get_unseen_test_object
from utils.metric import eval_object_pose, eval_batch_obj


def single_epoch(loader, model, epoch=None, optimizer=None, save_path="checkpoints",
                 train=True, save_results=False, indices_order=None, use_cuda=False):

    time_meters = AverageMeters()

    if train:
        print(f"training epoch: {epoch + 1}")
        avg_meters = AverageMeters()
        model.train()

    else:
        model.eval()

        # object evaluation
        REP_res_dict, ADD_res_dict= {}, {}
        diameter_dict = loader.dataset.obj_diameters
        mesh_dict = loader.dataset.obj_mesh
        mesh_dict, diameter_dict = filter_test_object(mesh_dict, diameter_dict)
        unseen_objects = get_unseen_test_object()
        for k in mesh_dict.keys():
            REP_res_dict[k] = []
            ADD_res_dict[k] = []

        if save_results:
            # save hand results for online evaluation
            xyz_pred_list, verts_pred_list = list(), list()

    end = time.time()
    for batch_idx, sample in enumerate(loader):
        if train:
            assert use_cuda and torch.cuda.is_available(), "requires cuda for training"
            imgs = sample["img"].float().cuda()
            bbox_hand = sample["bbox_hand"].float().cuda()
            bbox_obj = sample["bbox_obj"].float().cuda()

            mano_params = sample["mano_param"].float().cuda()
            joints_uv = sample["joints2d"].float().cuda()
            obj_p2d_gt = sample["obj_p2d"].float().cuda()
            obj_mask = sample["obj_mask"].float().cuda()

            # measure data loading time
            time_meters.add_loss_value("data_time", time.time() - end)
            # model forward
            model_loss, model_losses = model(imgs, bbox_hand, bbox_obj, mano_params=mano_params,
                                             joints_uv=joints_uv, obj_p2d_gt=obj_p2d_gt, obj_mask=obj_mask)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            for key, val in model_losses.items():
                if val is not None:
                    avg_meters.add_loss_value(key, val)

            # measure elapsed time
            time_meters.add_loss_value("batch_time", time.time() - end)

            # plot progress
            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s " \
                     "| Mano Mesh3D Loss: {mano_mesh3d_loss:.3f} " \
                     "| Mano Joints3D Loss: {mano_joints3d_loss:.3f} " \
                     "| Mano Shape Loss: {mano_shape_loss:.3f} | Mano Pose Loss: {mano_pose_loss:.3f} " \
                     "| Mano Total Loss: {mano_total_loss:.3f} | Heatmap Joints2D Loss: {hm_joints2d_loss:.3f} " \
                     "| Obj Reg Loss: {obj_reg_loss:.4f} | Obj conf Loss: {obj_conf_loss:.4f}" \
                     "| Total Loss: {total_loss:.3f} " \
                .format(batch=batch_idx + 1, size=len(loader),
                        data=time_meters.average_meters["data_time"].val,
                        bt=time_meters.average_meters["batch_time"].avg,
                        mano_mesh3d_loss=avg_meters.average_meters["mano_mesh3d_loss"].avg,
                        mano_joints3d_loss=avg_meters.average_meters["mano_joints3d_loss"].avg,
                        mano_shape_loss=avg_meters.average_meters["manoshape_loss"].avg,
                        mano_pose_loss=avg_meters.average_meters["manopose_loss"].avg,
                        mano_total_loss=avg_meters.average_meters["mano_total_loss"].avg,
                        hm_joints2d_loss=avg_meters.average_meters["hm_joints2d_loss"].avg,
                        obj_reg_loss=avg_meters.average_meters["obj_reg_loss"].avg,
                        obj_conf_loss=avg_meters.average_meters["obj_conf_loss"].avg,
                        total_loss=avg_meters.average_meters["total_loss"].avg)
            bar(suffix)
            end = time.time()

        else:
            if use_cuda and torch.cuda.is_available():
                imgs = sample["img"].float().cuda()
                bbox_hand = sample["bbox_hand"].float().cuda()
                bbox_obj = sample["bbox_obj"].float().cuda()
                if "root_joint" in sample:
                    root_joints = sample["root_joint"].float().cuda()
                else:
                    root_joints = None

            else:
                imgs = sample["img"].float()
                bbox_hand = sample["bbox_hand"].float()
                bbox_obj = sample["bbox_obj"].float()
                if "root_joint" in sample:
                    root_joints = sample["root_joint"].float()
                else:
                    root_joints = None

            # measure data loading time
            time_meters.add_loss_value("data_time", time.time() - end)

            preds_joints, results, preds_obj = model(imgs, bbox_hand, bbox_obj, roots3d=root_joints)
            pred_xyz = results["joints3d"].detach().cpu().numpy()
            pred_verts = results["verts3d"].detach().cpu().numpy()

            if save_results:
                for xyz, verts in zip(pred_xyz, pred_verts):
                    if indices_order is not None:
                        xyz = xyz[indices_order]
                    xyz_pred_list.append(xyz)
                    verts_pred_list.append(verts)

            # object predictions and evaluation(online)
            cam_intr = sample["cam_intr"].numpy()
            obj_pose = sample['obj_pose'].numpy()
            obj_cls = sample['obj_cls']
            obj_bbox3d = sample['obj_bbox3d'].numpy()
            REP_res_dict, ADD_res_dict = eval_batch_obj(preds_obj, bbox_obj,
                                                        obj_pose, mesh_dict, obj_bbox3d, obj_cls,
                                                        cam_intr, REP_res_dict, ADD_res_dict)
            # measure elapsed time
            time_meters.add_loss_value("batch_time", time.time() - end)

            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s"\
                .format(batch=batch_idx + 1, size=len(loader),
                        data=time_meters.average_meters["data_time"].val,
                        bt=time_meters.average_meters["batch_time"].avg)

            bar(suffix)
            end = time.time()

    if train:
        return avg_meters
    else:
        # object pose evaluation
        if REP_res_dict is not None and ADD_res_dict is not None \
                and diameter_dict is not None and unseen_objects is not None:
           eval_object_pose(REP_res_dict, ADD_res_dict, diameter_dict, outpath=save_path, unseen_objects=unseen_objects,
                            epoch=epoch+1 if epoch is not None else None)

        if save_results:
            pred_out_path = os.path.join(save_path, "pred_epoch_{}.json".format(epoch+1) if epoch is not None else "pred_{}.json")
            dump(pred_out_path, xyz_pred_list, verts_pred_list)
        return None