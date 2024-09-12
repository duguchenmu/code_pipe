from sift_lsd_demo.models.base_model import BaseModel
import numpy as np
from pytlsd import lsd
import torch
# import pycolmap
from .sift_keypoints import detect_sift_keypoint
from .line_concat import lines_to_wireframe

class TemplatePipeline():
    def detect_lsd_lines(self, x):
        lines, line_scores, valid_lines = [], [], []
        for b in range(len(x)):
            img = (x[b].squeeze().cpu().numpy() * 255).astype(np.uint8)
            for s in [0.3, 0.4, 0.5, 0.7, 0.8, 1.0]:
                b_segs = lsd(img, scale=s)
                if len(b_segs) >= 300:
                    break
        segs_length = np.linalg.norm(b_segs[:, 2:4] - b_segs[:, 0:2], axis=1)  # 计算长度
        # 排除短线段
        b_segs = b_segs[segs_length >= 15]
        segs_length = segs_length[segs_length >= 15]
        b_scores = b_segs[:, -1] * np.sqrt(segs_length)
        # 从高到底排列
        indices = np.argsort(-b_scores)
        # 选取前300个
        indices = indices[:300]
        lines.append(torch.from_numpy(b_segs[indices, :4].reshape(-1, 2, 2)))
        line_scores.append(torch.from_numpy(b_scores[indices]))
        valid_lines.append(torch.ones_like(line_scores[-1], dtype=torch.bool))
        lines = torch.stack(lines)
        line_scores = torch.stack(line_scores)
        valid_lines = torch.stack(valid_lines)
        return lines, line_scores, valid_lines

    def _forward(self, data):
        def process_siamese(data, i):
            data_i = {k[:-1]: v for k, v in data.items() if k[-1] == i}
            b_size, _, h, w = data_i['image'].shape
            #lsd line
            lines, line_scores, valid_lines = self.detect_lsd_lines(data_i['image'])
            if line_scores.shape[-1] != 0:
                line_scores /= (line_scores.new_tensor(1e-8) + line_scores.max(dim=1).values[:, None])
            # pred_lsd = {'lines': lines,
            #             'line_scores': line_scores,
            #             'valid_lines': valid_lines}
            #sift keypoint
            pred_sift = detect_sift_keypoint(data_i['image'])

            #删除过于接近线端点的关键点
            kp = pred_sift['keypoints']
            line_endpts = lines.reshape(b_size, -1, 2)
            dist_pt_lines = torch.norm(
                kp[:, :, None] - line_endpts[:, None], dim=-1)  # 关键点和线段端点之间的范数（距离）
            #对于每个关键点，将其标记为有效或要删除
            pts_to_remove = torch.any(
                dist_pt_lines < 4, dim=2)  # 维度 2 表示对每个点和线段的距离进行检查 返回布尔（至少有一个接近 就为T）
            #删除它们(这里我们假设batch_size = 1)
            assert len(kp) == 1
            pred_sift['keypoints'] = pred_sift['keypoints'][0][~pts_to_remove[0]][None]
            pred_sift['keypoint_scores'] = pred_sift['keypoint_scores'][0][~pts_to_remove[0]][None]
            pred_sift['descriptors'] = pred_sift['descriptors'][0].T[~pts_to_remove[0]].T[None]

            #把线连接在一起形成线框
            orig_lines = lines.clone()
            #首先合并相邻的端点以连接线路
            (line_points, line_pts_scores, line_descs, line_association,
             lines, lines_junc_idx, num_true_junctions) = lines_to_wireframe(
                lines, line_scores, pred_sift['all_descriptors'])
            #将关键点添加到连接点，并用随机关键点填充其余部分
            (all_points, all_scores, all_descs, pl_associativity) = [], [], [], []
            for bs in range(b_size):
                all_points.append(torch.cat(
                    [line_points[bs], pred_sift['keypoints'][bs]], dim=0))
                all_scores.append(torch.cat(
                    [line_pts_scores[bs], pred_sift['keypoint_scores'][bs]], dim=0))
                all_descs.append(torch.cat(
                    [line_descs[bs], pred_sift['descriptors'][bs]], dim=1))

                associativity = torch.eye(len(all_points[-1]), dtype=torch.bool)
                associativity[:num_true_junctions[bs], :num_true_junctions[bs]] = \
                    line_association[bs][:num_true_junctions[bs], :num_true_junctions[bs]]
                pl_associativity.append(associativity)
            all_points = torch.stack(all_points, dim=0)
            all_scores = torch.stack(all_scores, dim=0)
            all_descs = torch.stack(all_descs, dim=0)
            pl_associativity = torch.stack(pl_associativity, dim=0)
            del pred_sift['all_descriptors']
            torch.cuda.empty_cache()

            return {'keypoints': all_points,
                    'keypoint_scores': all_scores,
                    'descriptors': all_descs,
                    'pl_associativity': pl_associativity,
                    'num_junctions': torch.tensor(num_true_junctions),
                    'lines': lines,
                    'orig_lines': orig_lines,
                    'lines_junc_idx': lines_junc_idx,
                    'line_scores': line_scores,
                    'valid_lines': valid_lines}
            # return {**pred_sift, **pred_lsd}

        pred0 = process_siamese(data, '0')
        pred1 = process_siamese(data, '1')
        pred = {**{k + '0': v for k, v in pred0.items()},
                **{k + '1': v for k, v in pred1.items()}}


        return pred