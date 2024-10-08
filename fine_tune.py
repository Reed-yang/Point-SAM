import torch
import os
import numpy as np
import argparse
from demo.utils import load_ply
from point_sam import build_point_sam
import random
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from torch.utils.tensorboard import SummaryWriter
import os

# 参数解析
parser = argparse.ArgumentParser(description="PointSAM Command-Line Interface")
parser.add_argument("--checkpoint", type=str, default="pretrained/model.safetensors", help="Path to the model checkpoint.")
parser.add_argument("--pointcloud", type=str, default="yingrenshi_downsampled_10_100.txt", help="Path to the point cloud file (.ply).")
parser.add_argument("--output_dir", type=str, default="output/0906_bce_no_global_one_prompt", help="Directory to save the results.")
parser.add_argument("--iter", type=int, default=200, help="Number of fine-tuning iter.")
args = parser.parse_args()

# 全局变量
pc_xyz, pc_rgb = None, None
prompts, labels = [], []
prompt_mask = None
segment_mask = None
masks = []

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# 加载预训练模型并转为训练模式
sam = build_point_sam(args.checkpoint, num_group=2048, group_size=256).cuda()
sam.train()  # 设置模型为训练模式

# 损失函数和优化器
# criterion = nn.CrossEntropyLoss()  # 交叉熵损失
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(sam.mask_decoder.parameters(), lr=1e-4)  # 优化器

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    attributes = np.concatenate((xyz, rgb), axis=1)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def load_pointcloud(path):
    print(f"Loading point cloud from {path}...")
    points = load_ply(path)
    xyz = points[:, :3]
    rgb = points[:, 3:6] / 255

    # Normalize point cloud data
    shift = xyz.mean(0)
    scale = np.linalg.norm(xyz - shift, axis=-1).max()
    xyz = (xyz - shift) / scale

    return xyz, rgb

def load_pc_from_txt(path):
    print(f"Loading point cloud from {path}...")
    # points = load_ply(path)
    data = np.loadtxt(path, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                    ('semantic_label', 'i4'), ('instance_label', 'i4'),
                                    ('fine_grained_category', 'i4')])

    data_dict = {
        'xyz': np.column_stack([data['x'], data['y'], data['z']]),  # 合并 x, y, z 为一个数组
        'rgb': np.column_stack([data['red'], data['green'], data['blue']]),  # 合并 red, green, blue 为一个数组
        'semantic_label': data['semantic_label'],  # 直接提取 semantic_label
        'instance_label': data['instance_label'],  # 直接提取 instance_label
        'fine_grained_category': data['fine_grained_category']  # 直接提取 fine_grained_category
    }

    xyz = data_dict['xyz']
    rgb = data_dict['rgb'] / 255

    # Normalize point cloud data
    shift = xyz.mean(0)
    scale = np.linalg.norm(xyz - shift, axis=-1).max()
    xyz = (xyz - shift) / scale

    return xyz, rgb, data_dict

import torch

def buld_gt_mask_for_instance_seg(idx, data):
    """
    构建用于实例分割的ground truth mask。
    
    参数:
        xyz: (3,) 形状的 NumPy 数组，代表要进行分割的点的 x, y, z 坐标。
        data: 已加载的点云数据，包括 'semantic_label' 和 'instance_label' 信息。
    
    返回:
        mask: [total_points] 形状的布尔型张量，与输入的 xyz 同属于一个 semantic_label 和 instance_label 的点为 True，其他点为 False。
    """

    # 获取与该点相同的 semantic_label 和 instance_label
    semantic_label = data['semantic_label'][idx]
    instance_label = data['instance_label'][idx]

    # 根据 semantic_label 和 instance_label 构建布尔 mask
    mask = (data['semantic_label'] == semantic_label) & (data['instance_label'] == instance_label)

    # 将 mask 转换为 torch.bool 类型的张量
    mask_tensor = torch.tensor(mask, dtype=torch.bool)

    return mask_tensor

def buld_prompt_for_instance_seg(idx, data):
    ### 从给出的point idx 生成几个与其 semantic_label 和 instance_label 相同(属于同一个instance)的点作为prompt points， prompt label 为 1
    ## 从不属于 同一个instance 点中随机选取几个副样本点，prompt label 为 0

    positive_num = 3
    neg_num = 5

    # 获取与该点相同的 semantic_label 和 instance_label
    semantic_label = data['semantic_label'][idx]
    instance_label = data['instance_label'][idx]

    # 构建正样本 mask，筛选与该点属于同一个 semantic_label 和 instance_label 的点
    positive_mask = (data['semantic_label'] == semantic_label) & (data['instance_label'] == instance_label)

    # 获取所有符合条件的正样本点索引
    positive_indices = np.where(positive_mask)[0]

    # 如果正样本点少于需要的数量，直接全部选取
    if len(positive_indices) <= positive_num:
        selected_positive_indices = positive_indices
    else:
        # 随机选取正样本点
        selected_positive_indices = np.random.choice(positive_indices, size=positive_num, replace=False)

    # 构建负样本 mask，筛选与该点不属于同一个 instance 的点
    negative_mask = (data['semantic_label'] != semantic_label) | (data['instance_label'] != instance_label)

    # 获取所有符合条件的负样本点索引
    negative_indices = np.where(negative_mask)[0]

    # 如果负样本点少于需要的数量，直接全部选取
    if len(negative_indices) <= neg_num:
        selected_negative_indices = negative_indices
    else:
        # 随机选取负样本点
        selected_negative_indices = np.random.choice(negative_indices, size=neg_num, replace=False)

    # 合并正负样本点
    prompt_points = np.concatenate([selected_positive_indices, selected_negative_indices])

    # 构建对应的 labels：正样本为 1，负样本为 0
    prompt_labels = np.concatenate([np.ones(len(selected_positive_indices)), np.zeros(len(selected_negative_indices))])


    return prompt_points, prompt_labels


def segment_pointcloud(prompt_point, prompt_label):
    # global prompts, labels, prompt_mask, segment_mask
    # TODO global is a bug
    prompt_mask, segment_mask = None, None
    prompts, labels = [], []
    # Append prompt points and labels
    prompts.append(prompt_point)
    labels.append(prompt_label)

    # Prepare data for model
    # prompt_points = torch.from_numpy(np.array(prompts)).cuda().float()[None, ...]
    prompt_points = torch.from_numpy(np.array(prompts)).cuda().float().reshape(1, -1, 3)
    # prompt_labels = torch.from_numpy(np.array(labels)).cuda()[None, ...]
    prompt_labels = torch.from_numpy(np.array(labels)).cuda().float().reshape(1, -1)

    sam.set_pointcloud(pc_xyz, pc_rgb)
    mask, scores, logits = sam.predict_masks(
        prompt_points, prompt_labels, prompt_mask, prompt_mask is None
    )

    # Update the mask
    prompt_mask = logits[0][torch.argmax(scores[0])][None, ...]
    segment_mask = mask[0][torch.argmax(scores[0])] > 0

    return segment_mask, scores, logits

def save_results():
    os.makedirs(args.output_dir, exist_ok=True)
    global pc_xyz, pc_rgb, segment_mask, masks

    xyz = pc_xyz[0].cpu().numpy()
    rgb = pc_rgb[0].cpu().numpy()
    masks = np.stack(masks)

    # Save the results as a .npy file
    np.save(f"{args.output_dir}/segment_results.npy", {"xyz": xyz, "rgb": rgb, "mask": masks})
    print(f"Results saved to {args.output_dir}/segment_results.npy")

def fine_tune():
    # Load the point cloud and ground truth labels
    global pc_xyz, pc_rgb

    if args.pointcloud.endswith("ply"):
        pc_xyz, pc_rgb = load_pointcloud(args.pointcloud)
    elif args.pointcloud.endswith("txt"):
        pc_xyz, pc_rgb, data = load_pc_from_txt(args.pointcloud)
    else:
        raise NotImplementedError("point cloud format not supported yet.")
    pc_xyz, pc_rgb = torch.from_numpy(pc_xyz).cuda().float(), torch.from_numpy(pc_rgb).cuda().float()
    pc_xyz, pc_rgb = pc_xyz.unsqueeze(0), pc_rgb.unsqueeze(0)

    total_points = pc_xyz.shape[1]
    writer = SummaryWriter(log_dir=args.output_dir)

    # 创建 tqdm 进度条实例
    progress_bar = tqdm(range(args.iter), desc="Fine-tuning", ncols=100)

    # 开始fine-tuning循环
    for iter in progress_bar:
        optimizer.zero_grad()  # 清空优化器的梯度

        # TODO
        # only pick building point for training
        # now_pick_idx, now_pick_xyz, gt_mask = building_point_for_training()

        # 从点云中随机选择一个点
        now_pick_idx = random.randint(0, total_points - 1)
        now_pick_xyz = pc_xyz[0, now_pick_idx]

        gt_mask = buld_gt_mask_for_instance_seg(now_pick_idx, data)

        # 使用 ground truth 的标签
        prompt_label = 1 # TODO always positive prompt
        prompt_point = np.array(now_pick_xyz.detach().cpu())

        # prompt_point_idx, prompt_label__ = buld_prompt_for_instance_seg(now_pick_idx, data)
        # prompt_label_ = np.ones(1)
        # prompt_label = np.concatenate([prompt_label_, prompt_label__]).astype(int)
        # prompt_point = np.concatenate([pc_xyz[0, now_pick_idx].unsqueeze(0).cpu(), pc_xyz[0, prompt_point_idx].cpu()])

        # 调用 segmentation 函数
        mask, scores, logits = segment_pointcloud(prompt_point, prompt_label)

        # Ground truth 和 logit 的交叉熵损失
        # gt_mask_tensor = torch.tensor([gt_mask], dtype=torch.long).cuda()
        gt_mask_tensor = gt_mask.float().cuda()
        # loss = criterion(logits[0].view(1, -1), gt_mask_tensor)
        pred = logits[0][torch.argmax(scores[0])]
        loss = criterion(pred, gt_mask_tensor)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # Log loss to TensorBoard
        writer.add_scalar("Loss/train", loss.item(), iter)
        progress_bar.set_postfix(loss=loss.item())

        # log seg-ed pc and gt pc
        if iter % 3 == 0:
            gt_xyz, gt_rgb = pc_xyz[0][gt_mask], pc_rgb[0][gt_mask]
            storePly(f"{args.output_dir}/gt_pc_{iter}.ply", gt_xyz.cpu(), gt_rgb.cpu() * 255)
            
            seg_xyz, seg_rgb = pc_xyz[0][mask], pc_rgb[0][mask]
            storePly(f"{args.output_dir}/seg_pc_{iter}.ply", seg_xyz.cpu(), seg_rgb.cpu() * 255)

    # Close TensorBoard writer
    writer.close()
    # 保存最终结果
    save_results()

if __name__ == "__main__":
    fine_tune()
