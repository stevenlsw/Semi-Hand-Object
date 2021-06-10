import numpy as np
import os
import cv2

# object evaluation metric
# hand evaluation metric: https://github.com/shreyashampali/ho3d/blob/master/eval.py


def vertices_reprojection(vertices, rt, k):
    p = np.matmul(k, np.matmul(rt[:3, 0:3], vertices.T) + rt[:3, 3].reshape(-1, 1))
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T


def compute_REP_error(pred_pose, gt_pose, intrinsics, obj_mesh):
    reproj_pred = vertices_reprojection(obj_mesh, pred_pose, intrinsics)
    reproj_gt = vertices_reprojection(obj_mesh, gt_pose, intrinsics)
    reproj_diff = np.abs(reproj_gt - reproj_pred)
    reproj_bias = np.mean(np.linalg.norm(reproj_diff, axis=1), axis=0)
    return reproj_bias


def compute_ADD_error(pred_pose, gt_pose, obj_mesh):
    add_gt = np.matmul(gt_pose[:3, 0:3], obj_mesh.T) + gt_pose[:3, 3].reshape(-1, 1)  # (3,N)
    add_pred = np.matmul(pred_pose[:3, 0:3], obj_mesh.T) + pred_pose[:3, 3].reshape(-1, 1)
    add_bias = np.mean(np.linalg.norm(add_gt - add_pred, axis=0), axis=0)
    return add_bias


def fuse_test(output, width, height, intrinsics, bestCnt, bbox_3d, cord_upleft, affinetrans=None):
    predx = output[0]
    predy = output[1]
    det_confs = output[2]
    keypoints = bbox_3d
    nH, nW, nV = predx.shape

    xs = predx.reshape(nH * nW, -1) * width
    ys = predy.reshape(nH * nW, -1) * height
    det_confs = det_confs.reshape(nH * nW, -1)
    gridCnt = len(xs)

    p2d = None
    p3d = None
    candiBestCnt = min(gridCnt, bestCnt)
    for i in range(candiBestCnt):
        bestGrids = det_confs.argmax(axis=0) # choose best N count
        validmask = (det_confs[bestGrids, list(range(nV))] > 0.5)
        xsb = xs[bestGrids, list(range(nV))][validmask]
        ysb = ys[bestGrids, list(range(nV))][validmask]
        t2d = np.concatenate((xsb.reshape(-1, 1), ysb.reshape(-1, 1)), 1)
        t3d = keypoints[validmask]
        if p2d is None:
            p2d = t2d
            p3d = t3d
        else:
            p2d = np.concatenate((p2d, t2d), 0)
            p3d = np.concatenate((p3d, t3d), 0)
        det_confs[bestGrids, list(range(nV))] = 0

    if len(p3d) < 6:
        R = np.eye(3)
        T = np.array([0, 0, 1]).reshape(-1, 1)
        rt = np.concatenate((R, T), 1)
        return rt, p2d

    p2d[:, 0] += cord_upleft[0]
    p2d[:, 1] += cord_upleft[1]
    if affinetrans is not None:
        homp2d = np.concatenate([p2d, np.ones([np.array(p2d).shape[0], 1])], 1)
        p2d = affinetrans.dot(homp2d.transpose()).transpose()[:, :2]
    retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)
    if not retval:
        R = np.eye(3)
        T = np.array([0, 0, 1]).reshape(-1, 1)
    else:
        R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        T = trans.reshape(-1, 1)
    rt = np.concatenate((R, T), 1)
    return rt, p2d


def eval_batch_obj(batch_output, obj_bbox,
                   obj_pose, mesh_dict, obj_bbox3d, obj_cls,
                   cam_intr, REP_res_dic, ADD_res_dic, bestCnt=10, batch_affinetrans=None):
    # bestCnt: choose best N count for fusion
    bs = batch_output[0].shape[0]
    obj_bbox = obj_bbox.cpu().numpy()
    for i in range(bs):
        output = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        bbox = obj_bbox[i]
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cord_upleft = [bbox[0], bbox[1]]
        intrinsics = cam_intr[i]
        bbox_3d = obj_bbox3d[i]
        cls = obj_cls[i]
        mesh = mesh_dict[cls]
        if batch_affinetrans is not None:
            affinetrans = batch_affinetrans[i]
        else:
            affinetrans = None
        pred_pose, p2d = fuse_test(output, width, height, intrinsics, bestCnt, bbox_3d, cord_upleft,
                                   affinetrans=affinetrans)
        # calculate REP and ADD error
        REP_error = compute_REP_error(pred_pose, obj_pose[i], intrinsics, mesh)
        ADD_error = compute_ADD_error(pred_pose, obj_pose[i], mesh)
        REP_res_dic[cls].append(REP_error)
        ADD_res_dic[cls].append(ADD_error)
    return REP_res_dic, ADD_res_dic


def eval_object_pose(REP_res_dic, ADD_res_dic, diameter_dic, outpath, unseen_objects=[], epoch=None):
    # REP_res_dic: key: object class, value: REP error distance
    # ADD_res_dic: key: object class, value: ADD error distance

    # object result file
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    log_path = os.path.join(outpath, "object_result.txt") if epoch is None else os.path.join(outpath, "object_result_epoch{}.txt".format(epoch))
    log_file = open(log_path, "w+")

    REP_5 = {}
    for k in REP_res_dic.keys():
        REP_5[k] = np.mean(np.array(REP_res_dic[k]) <= 5)

    ADD_10 = {}
    for k in ADD_res_dic.keys():
        ADD_10[k] = np.mean(np.array(ADD_res_dic[k]) <= 0.1 * diameter_dic[k])

    for k in ADD_res_dic.keys():
        if k in unseen_objects:
            REP_5.pop(k, None)
            ADD_10.pop(k, None)

    # write down result
    print('REP-5', file=log_file)
    print(REP_5, file=log_file)
    print('ADD-10', file=log_file)
    print(ADD_10, file=log_file)
    log_file.close()
    return ADD_10, REP_5