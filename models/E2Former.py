import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trans import Transformer as Transformer_encoder
from einops import rearrange, repeat
import torch_dct as dct



def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density

        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


class E2foremr(nn.Module):
    def __init__(self, eeg_channel=32, eye_channel=4, num_classes=13):
        super(E2foremr, self).__init__()
        self.eeg_cnn = nn.Sequential(
            nn.Conv1d(eeg_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.eye_cnn = nn.Sequential(
            nn.Conv1d(eye_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fusion_fc = nn.Linear(128 + 64, 128)
        self.transformer = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=2)
        self.dropout = nn.Dropout(0.1)  # 添加 Dropout
        self.fc = nn.Linear(128, num_classes)

    def forward(self, eye_data, eeg_data):
        # 插值对齐：将眼动数据从600插值到1280
        eye_data = F.interpolate(eye_data.permute(0, 2, 1), size=1280, mode='linear', align_corners=False)
        eye_data = eye_data.permute(0, 2, 1)  # 恢复形状 (b, 1280, 4)

        # EEG特征提取
        eeg_features = self.eeg_cnn(eeg_data.permute(0, 2, 1))  # (b, 128, 320)
        eeg_features = eeg_features.permute(0, 2, 1)  # (b, 320, 128)

        # 眼动特征提取
        eye_features = self.eye_cnn(eye_data.permute(0, 2, 1))  # (b, 64, 320)
        eye_features = eye_features.permute(0, 2, 1)  # (b, 320, 64)

        # 多模态融合
        combined = torch.cat((eeg_features, eye_features), dim=2)  # (b, 320, 192)
        combined = self.fusion_fc(combined)  # (b, 320, 128)

        # 时序建模
        combined = combined.permute(1, 0, 2)  # (320, b, 128)
        transformer_output = self.transformer(combined, combined)  # (320, b, 128)
        transformer_output = transformer_output.permute(1, 0, 2)  # (b, 320, 128)

        transformer_output = self.dropout(transformer_output)

        # 分类
        output = self.fc(transformer_output.mean(dim=1))  # (b, num_classes)
        return output

