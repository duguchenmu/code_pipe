from sklearn.cluster import DBSCAN
import torch

def sample_descriptors(keypoints, descriptors, s):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args) #对 descriptors 张量进行了网格采样操作，根据给定的关键点位置 keypoints  ，采用双线性插值方法
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors
def lines_to_wireframe(lines, line_scores, all_descs):
    b_size, _, _, _ = all_descs.shape
    device = lines.device
    endpoints = lines.reshape(b_size, -1, 2)
    (junctions, junc_scores, junc_descs, connectivity, new_lines,
     lines_junc_idx, num_true_junctions) = [], [], [], [], [], [], []
    for bs in range(b_size):
        #把附近的结点聚在一起
        db = DBSCAN(eps=3, min_samples=1).fit(endpoints[bs].cpu().numpy())
        clusters = db.labels_
        n_clusters = len(set(clusters))
        num_true_junctions.append(n_clusters)
        #计算每个簇的平均连接数和得分
        clusters = torch.tensor(clusters, dtype=torch.long, device=device)
        new_junc = torch.zeros(n_clusters, 2, dtype=torch.float, device=device)
        new_junc.scatter_reduce_(0, clusters[:, None].repeat(1, 2), endpoints[bs], reduce='mean', include_self=False)
        junctions.append(new_junc)
        new_scores = torch.zeros(n_clusters, dtype=torch.float, device=device)
        new_scores.scatter_reduce_(0, clusters, torch.repeat_interleave(line_scores[bs], 2), reduce='mean', include_self=False)
        junc_scores.append(new_scores)

        # Compute the new lines
        new_lines.append(junctions[-1][clusters].reshape(-1, 2, 2))  # 是根据聚类结果选择特定的聚类中心，并按照指定的形状重新组织后的结果，将其添加到 new_lines 列表中
        lines_junc_idx.append(clusters.reshape(-1, 2))

        #计算节点连通度
        junc_connect = torch.eye(n_clusters, dtype=torch.bool, device=device)  # 形状为 (n_clusters, n_clusters) 的单位矩阵
        pairs = clusters.reshape(-1, 2)  # these pairs are connected by a line
        junc_connect[pairs[:, 0], pairs[:, 1]] = True
        junc_connect[pairs[:, 1], pairs[:, 0]] = True
        connectivity.append(junc_connect)

        # Interpolate the new junction descriptors 插值新的连接描述子
        junc_descs.append(sample_descriptors(junctions[-1][None], all_descs[bs:(bs + 1)], 8)[0])

    new_lines = torch.stack(new_lines, dim=0)
    lines_junc_idx = torch.stack(lines_junc_idx, dim=0)
    return (junctions, junc_scores, junc_descs, connectivity,
            new_lines, lines_junc_idx, num_true_junctions)