
import torch
from torch import nn
import numpy as np
import cv2
import itertools
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, nms_gpu
from sklearn.neighbors import NearestNeighbors
# from vlib.point import draw_points_boxes_plt, draw_box_plt
import matplotlib.pyplot as plt
from opencood.utils.max_consensus import max_consunsus_hierarchical

pi = 3.141592653


def limit_period(val, offset=0.5, period=2 * pi):
    return val - torch.floor(val / period + offset) * period


class Matcher(nn.Module):
    """Correct localization error and use Algorithm 1: BBox matching with scores to fuse the proposal BBoxes"""
    def __init__(self, cfg, pc_range, has_noise=False, search_range=None):
        super(Matcher, self).__init__()
        self.pc_range = pc_range
        self.has_noise = has_noise
        if self.has_noise:
            assert 'max_cons' in cfg and 'search_range' in cfg, \
                "parameters for maximum consensus algorithm must be configured if localization noise exists"
            self.max_cons_cfg = cfg['max_cons']
            self.max_cons_cfg['search_range'] = search_range

    @torch.no_grad()
    def forward(self, data_dict):
        tfs = data_dict['translations']
        if self.has_noise:
            data_dict = self.err_correction(data_dict)
        else:
            boxes_ego_cs = []
            for i, boxes in enumerate(data_dict['det_boxes']):
                boxes[:, :2] = boxes[:, :2] + tfs[i, :2] - tfs[0, :2]
                boxes_ego_cs.append(boxes)
                if 'point_coords' in data_dict:
                    data_dict['point_coords'][i][:, :2] = data_dict['point_coords'][i][:, :2] + tfs[i, :2] - tfs[0, :2]

            data_dict['det_boxes_ego_coords'] = boxes_ego_cs
            if 'point_coords' in data_dict:
                data_dict['cpm_pts_features'] = data_dict['point_features']
                data_dict['cpm_pts_coords'] = data_dict['point_coords']

        # Ts_gt = data_dict['errs_T']
        data_dict['boxes_fused'], data_dict['scores_fused'] = self.fusion(data_dict)

        clouds = data_dict['points']
        clouds_fused = [clouds[clouds[:, 0]==i, 1:] for i in range(len(tfs))]
        Ts = data_dict['err_T_est'] if self.has_noise else []
        for i in range(1, len(tfs)):
            # clouds_fused[i][:, :2] = clouds_fused[i][:, :2] - tfs[0][:2] + tfs[i][:2]
            if self.has_noise and Ts[i-1] is not None:
                # correct the coop point clouds
                tmp = torch.cat([clouds_fused[i][:, :2], torch.ones((len(clouds_fused[i][:, :2]), 1),
                                 device=clouds_fused[i][:, :2].device)], dim=1)
                clouds_fused[i][:, :2] = (Ts[i-1] @ tmp.T)[:2, :].T
            else:
                clouds_fused[i][:, :2] = clouds_fused[i][:, :2] + tfs[i, :2] - tfs[0, :2]
        data_dict['points_fused'] = clouds_fused

        return data_dict

    def fusion(self, data_dict):
        """
        Assign predicted boxes to clusters according to their ious with each other
        """
        pred_scores = data_dict['det_scores']
        pred_boxes_cat = torch.cat(data_dict['det_boxes_ego_coords'], dim=0)
        pred_scores_cat = torch.cat(pred_scores, dim=0)
        # if torch.isnan(pred_boxes_cat).any():
        #     print('debug')

        mask = nms_gpu(pred_boxes_cat, pred_scores_cat, thresh=0.01)[0]

        return pred_boxes_cat[mask], pred_scores_cat[mask]

    def cluster_fusion(self, clusters, scores):
        """
        Merge boxes in each cluster with scores as weights for merging
        """
        boxes_fused = []
        scores_fused = []
        for c, s in zip(clusters, scores):
            # reverse direction for non-dominant direction of boxes
            dirs = c[:, -1]
            max_score_idx = torch.argmax(s)
            dirs_diff = torch.abs(dirs - dirs[max_score_idx].item())
            lt_pi = (dirs_diff > pi).int()
            dirs_diff = dirs_diff * (1 - lt_pi) + (2 * pi - dirs_diff) * lt_pi
            score_lt_half_pi = s[dirs_diff > pi / 2].sum() # larger than
            score_set_half_pi = s[dirs_diff <= pi / 2].sum() # small equal than
            # select larger scored direction as final direction
            if score_lt_half_pi <= score_set_half_pi:
                dirs[dirs_diff > pi / 2] += pi
            else:
                dirs[dirs_diff <= pi / 2] += pi
            dirs = limit_period(dirs)
            s_normalized = s / s.sum()
            sint = torch.sin(dirs) * s_normalized
            cost = torch.cos(dirs) * s_normalized
            theta = torch.atan2(sint.sum(), cost.sum()).view(1,)
            center_dim = c[:, :-1] * s_normalized[:, None]
            boxes_fused.append(torch.cat([center_dim.sum(dim=0), theta]))
            s_sorted = torch.sort(s, descending=True).values
            s_fused = 0
            for i, ss in enumerate(s_sorted):
                s_fused += ss**(i+1)
            s_fused = torch.tensor([min(s_fused, 1.0)], device=s.device)
            scores_fused.append(s_fused)
        if len(boxes_fused) > 0:
            boxes_fused = torch.stack(boxes_fused, dim=0)
            scores_fused = torch.stack(scores_fused, dim=0)
        else:
            boxes_fused = None
            scores_fused = None
            print('debug')
        return boxes_fused, scores_fused

    def err_correction(self, data_dict):
        pred_boxes = data_dict['det_boxes']
        # kpts_coords = data_dict.get('point_coords', None)
        tfs = data_dict['translations']
        pts_feat = data_dict['cpm_pts_features']
        pts_coords = data_dict['cpm_pts_coords']
        pts_cls = data_dict['cpm_pts_cls']

        T_list = []
        tf_local_list = []
        corrected_boxes_list = [pred_boxes[0]]
        corrected_points_list = [pts_coords[0]]
        for boxes, points, tf, lbls in zip(pred_boxes[1:], pts_coords[1:], tfs[1:], pts_cls[1:]):
            T, tf_local = self.matching(pred_boxes[0], boxes, pts_coords[0],
                              points, tfs[0:1], tf.reshape(1, -1), pts_cls[0], lbls)
            if T is not None:
                # correct coords of points2
                tmp = torch.cat([points[:, :2], torch.ones((len(points), 1), device=points.device)], dim=1)
                points[:, :2] = (T @ tmp.T)[:2, :].T
                # correct coords of boxes2
                tmp = torch.cat([boxes[:, :2], torch.ones((len(boxes), 1), device=boxes.device)], dim=1)
                # cur_boxes[:, :2] = (tmp[:, :3] @ T)[: , :2]
                boxes[:, :2] = (T @ tmp.T)[:2, :].T
                boxes[:, -1] += torch.atan2(T[1, 0], T[0, 0])#tf_local[2]
            else:
                points[:, :2] = points[:, :2] + tf[:2] + tfs[0, :2]
                boxes[:, :2] = boxes[:, :2] + tf[:2] + tfs[0, :2]

            T_list.append(T)
            tf_local_list.append(tf_local)
            corrected_boxes_list.append(boxes)
            corrected_points_list.append(points)

        data_dict['det_boxes_ego_coords'] = corrected_boxes_list
        data_dict['cpm_pts_coords'] = corrected_points_list
        data_dict['err_T_est'] = T_list
        data_dict['err_tf_est_local'] = tf_local_list
        return data_dict

    def matching(self, boxes1, boxes2, points1, points2, loc1, loc2, lbls1, lbls2):
        """
        register boxes2 to boxes 1
        """
        ego_bbox_mask = torch.norm(boxes1[:, :2] + loc1[:, :2] - loc2[:, :2], dim=1) < 57.6
        coop_bbox_mask = torch.norm(boxes2[:, :2] + loc2[:, :2] - loc1[:, :2], dim=1) < 57.6
        ego_boxes_masked = boxes1[ego_bbox_mask]
        coop_boxes_masked = boxes2[coop_bbox_mask]

        cls_mask1 = torch.logical_and(lbls1>0, lbls1<4)
        cls_mask2 = torch.logical_and(lbls2>0, lbls2<4)
        points1_s = points1[cls_mask1]
        points2_s = points2[cls_mask2]
        mask1 = torch.norm(points1_s[:, :2] + loc1[:, :2] - loc2[:, :2], dim=1) < 57.6
        mask2 = torch.norm(points2_s[:, :2] + loc2[:, :2] - loc1[:, :2], dim=1) < 57.6
        dst = torch.cat([ego_boxes_masked[:, :2], points1_s[mask1, :2]], dim=0).cpu().numpy()
        src = torch.cat([coop_boxes_masked[:, :2], points2_s[mask2, :2]], dim=0).cpu().numpy()
        labels1 = np.concatenate([np.ones(len(ego_boxes_masked), dtype=np.int64) * 4, lbls1[cls_mask1][mask1].cpu().numpy()], axis=0)
        labels2 = np.concatenate([np.ones(len(coop_boxes_masked), dtype=np.int64) * 4, lbls2[cls_mask2][mask2].cpu().numpy()], axis=0)
        T, tf_local, _ = max_consunsus_hierarchical(dst, src, loc1.cpu().numpy(), loc2.cpu().numpy(),
                                                    point_labels=(labels1, labels2),
                                                    label_weights=[0, 1, 1, 2, 2], **self.max_cons_cfg)
        if T is None:
            return None, None
        T = torch.tensor(T, device=boxes1.device, dtype=boxes1.dtype)
        tf_local = torch.tensor(tf_local, device=boxes1.device, dtype=boxes1.dtype)
        return T, tf_local

    def estimate_tf_2d(self, pointsl, pointsr):
        is_numpy = False
        if not isinstance(pointsl, torch.Tensor):
            pointsl = torch.tensor(pointsl)
            pointsr = torch.tensor(pointsr)
            is_numpy = True
        # 1 reduce by the center of mass
        l_mean = pointsl.mean(dim=0)
        r_mean = pointsr.mean(dim=0)
        l_reduced = pointsl - l_mean
        r_reduced = pointsr - r_mean
        # 2 compute the rotation
        Sxx = (l_reduced[:, 0] * r_reduced[:, 0]).sum()
        Syy = (l_reduced[:, 1] * r_reduced[:, 1]).sum()
        Sxy = (l_reduced[:, 0] * r_reduced[:, 1]).sum()
        Syx = (l_reduced[:, 1] * r_reduced[:, 0]).sum()
        theta = torch.atan2(Sxy - Syx, Sxx + Syy)  # / np.pi * 180
        t = r_mean.reshape(2, 1) - torch.tensor([[torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]], device=theta.device) @ l_mean.reshape(2, 1)
        if is_numpy:
            theta = theta.cpu().numpy()
            t = t.cpu().numpy()
        return theta, t.T

    def rotate_points_along_z(self, points, angle):
        """
        Args:
            points: (N, 2 + C)
            angle: float, angle along z-axis, angle increases x ==> y
        Returns:

        """
        out_dim = points.shape[1]
        if out_dim==2:
            points = torch.cat([points, torch.zeros((points.shape[0], 1), device=points.device)], dim=-1)
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        rot_matrix = torch.tensor([
            [cosa, -sina, 0],
            [sina, cosa, 0],
            [0, 0, 1]
        ]).float().to(angle.device)
        points_rot = torch.matmul(points[:, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, 3:]), dim=-1)
        points_out = points_rot[:, :2] if out_dim==2 else points_rot
        return points_out

    def tfs_to_Tmat(self, Ts):
        """
        Convert a list of transformations to one transformation matrix Tmat
        Ts: list of [dx, dy, theta]
        """
        Tmat = torch.eye(3).to(Ts[0].device)
        for T in Ts:
            cosa = torch.cos(T[2])
            sina = torch.sin(T[2])
            rot_matrix = torch.tensor([
                [cosa, -sina, T[0]],
                [sina, cosa, T[1]],
                [0, 0, 1]
            ]).float().to(T.device)
            Tmat = rot_matrix @ Tmat
        return Tmat



