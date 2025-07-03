# 文件名: train_conceptual_mps.py
# 目的: 支持 MPS 加速的 3DGS 训练脚本，适用于 Apple Silicon 设备
#
# 警告: 此脚本为教育目的，实际训练仍可能较慢
#
# 前置要求:
# 1. macOS 12.3+ 和 Apple Silicon (M1/M2/M3) 芯片
# 2. 已安装 PyTorch 2.0+，支持 MPS
# 3. 其他依赖: pip install numpy Pillow tqdm pycolmap scipy plyfile

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from plyfile import PlyData
import pycolmap
from pathlib import Path
import time
import math

# 设置设备 (优先使用 MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# --- 1. 数据加载辅助函数 ---

def load_colmap_data(source_path, device=device):
    """
    加载由官方 convert.py 处理过的COLMAP项目文件夹
    """
    source_path = Path(source_path)
    sparse_path = source_path / "sparse" / "0"

    if not sparse_path.exists():
        raise FileNotFoundError(f"错误: 找不到稀疏模型文件夹 '{sparse_path}'")

    print(f"加载稀疏模型: '{sparse_path}'...")
    reconstruction = pycolmap.Reconstruction(sparse_path)
    
    cameras_colmap = reconstruction.cameras
    images_colmap = reconstruction.images
    
    cameras = []
    for image_id, image in images_colmap.items():
        cam_params = cameras_colmap[image.camera_id]
        
        pose_w2c = image.cam_from_world()
        R_w2c_np = pose_w2c.rotation.matrix()
        tvec_w2c_np = pose_w2c.translation
        
        # 将数据直接放在目标设备上
        R_w2c = torch.from_numpy(R_w2c_np).float().to(device)
        tvec_w2c = torch.from_numpy(tvec_w2c_np).float().to(device)
        
        R_c2w = R_w2c.transpose(0, 1)
        t_c2w = -torch.matmul(R_c2w, tvec_w2c)
        
        cam = {
            'id': image.image_id,
            'img_name': image.name,
            'width': cam_params.width,
            'height': cam_params.height,
            'position': t_c2w,
            'R_w2c': R_w2c,
            'tvec_w2c': tvec_w2c,
            'fy': torch.tensor(cam_params.params[1], dtype=torch.float32).to(device),
            'fx': torch.tensor(cam_params.params[0], dtype=torch.float32).to(device),
            'cx': torch.tensor(cam_params.width / 2, dtype=torch.float32).to(device),
            'cy': torch.tensor(cam_params.height / 2, dtype=torch.float32).to(device)
        }
        cameras.append(cam)
    
    # 加载初始点云
    points3D = reconstruction.points3D
    points = np.array([p.xyz for p in points3D.values()])
    colors = np.array([p.color for p in points3D.values()]) / 255.0
    
    return (
        cameras, 
        torch.tensor(points, dtype=torch.float32).to(device), 
        torch.tensor(colors, dtype=torch.float32).to(device)
    )

# --- 2. 核心：高斯模型与渲染 (MPS优化) ---

class GaussianModel:
    def __init__(self, initial_points, initial_colors, device=device):
        self.device = device
        self.means = torch.nn.Parameter(initial_points.clone().to(device))
        self.colors = torch.nn.Parameter(initial_colors.clone().to(device))
        num_points = initial_points.shape[0]
        
        # 初始化参数
        self.scales = torch.nn.Parameter(
            torch.log(torch.ones(num_points, 3, device=device) * 0.01))
        
        # 四元数 (w, x, y, z) 初始化为无旋转
        self.quats = torch.nn.Parameter(torch.cat([
            torch.ones(num_points, 1, device=device), 
            torch.zeros(num_points, 3, device=device)
        ], dim=1))
        
        # 不透明度 (使用logit变换)
        self.opacities = torch.nn.Parameter(
            torch.logit(torch.ones(num_points, 1, device=device) * 0.1))
        
        # 用于记录梯度信息
        self.max_grad = 0.0
        self.percent_dense = 0.01

    def get_params(self):
        return [self.means, self.colors, self.scales, self.quats, self.opacities]

def quat_to_rot_matrix(q):
    """将四元数转换为旋转矩阵 (MPS兼容)"""
    w, x, y, z = q.unbind(dim=-1)
    
    # 计算旋转矩阵元素
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    rot_matrix = torch.stack([
        1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy),
        2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)
    ], dim=-1).view(-1, 3, 3)
    
    return rot_matrix

def render_gaussians(camera, gaussians):
    """优化的渲染器，修复了边界问题"""
    width, height = int(camera['width']), int(camera['height'])
    fx, fy = camera['fx'], camera['fy']
    cx, cy = camera['cx'], camera['cy']
    
    R_w2c = camera['R_w2c']
    tvec_w2c = camera['tvec_w2c']
    
    # 1. 将点云从世界坐标系变换到相机坐标系
    means_cam = (gaussians.means - tvec_w2c) @ R_w2c.T
    
    # 2. 投影到图像平面
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    # 3. 计算每个高斯的半径
    scales = torch.exp(gaussians.scales)
    avg_scales = torch.mean(scales, dim=1)
    radii = (3 * avg_scales * min(fx, fy) / z).int().clamp(min=1)
    
    # 4. 初始化渲染图像和深度缓冲区
    rendered_image = torch.zeros(height, width, 3, device=gaussians.device)
    z_buffer = torch.full((height, width), float('inf'), device=gaussians.device)
    
    # 5. 按深度排序 (从远到近)
    sorted_indices = torch.argsort(z, descending=True)
    
    # 6. 渲染循环 (修复边界问题)
    for idx in sorted_indices:
        zi = z[idx]
        if zi <= 0.1:  # 跳过相机后面的点
            continue
            
        center_u, center_v = u[idx], v[idx]
        opacity = torch.sigmoid(gaussians.opacities[idx])
        color = gaussians.colors[idx]
        radius = radii[idx].item()
        
        # 计算影响区域，确保区域有效
        min_u = max(0, int(center_u - radius))
        max_u = min(width, int(center_u + radius + 1))
        min_v = max(0, int(center_v - radius))
        max_v = min(height, int(center_v + radius + 1))
        
        # 检查区域是否有效
        if min_u >= max_u or min_v >= max_v:
            continue
        
        # 创建像素网格
        px = torch.arange(min_u, max_u, device=gaussians.device, dtype=torch.float32)
        py = torch.arange(min_v, max_v, device=gaussians.device, dtype=torch.float32)
        
        # 创建网格并计算距离
        grid_x, grid_y = torch.meshgrid(px, py, indexing='xy')
        dx = grid_x - center_u
        dy = grid_y - center_v
        dist_sq = dx**2 + dy**2
        
        # 计算高斯权重
        sigma = max(radius / 3.0, 1e-3)  # 避免除零错误
        weights = torch.exp(-dist_sq / (2 * sigma**2))
        
        # 深度测试
        valid_mask = (zi < z_buffer[min_v:max_v, min_u:max_u])
        
        # Alpha混合
        alpha = opacity * weights
        for c in range(3):
            channel_slice = rendered_image[min_v:max_v, min_u:max_u, c]
            channel_slice[valid_mask] = (
                color[c] * alpha[valid_mask] + 
                channel_slice[valid_mask] * (1 - alpha[valid_mask]))
        
        # 更新深度缓冲区
        z_buffer[min_v:max_v, min_u:max_u][valid_mask] = zi
    
    return rendered_image.permute(2, 0, 1)

# --- 3. 主训练逻辑 ---

def train(args):
    print(f"--- 开始训练 (设备: {device}) ---")
    
    # 1. 加载数据
    print("加载COLMAP数据...")
    cameras, initial_points, initial_colors = load_colmap_data(args.source_path, device)
    print(f"加载了 {len(cameras)} 个相机视角和 {len(initial_points)} 个初始点")
    
    # 2. 初始化高斯模型和优化器
    gaussians = GaussianModel(initial_points, initial_colors, device)
    optimizer = torch.optim.Adam(gaussians.get_params(), lr=args.learning_rate)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # 3. 训练循环
    for iteration in range(1, args.iterations + 1):
        cam_idx = np.random.randint(0, len(cameras))
        viewpoint_cam = cameras[cam_idx]
        
        gt_image_path = os.path.join(args.source_path, "input", viewpoint_cam['img_name'])
        try:
            gt_image = Image.open(gt_image_path)
            gt_tensor = torch.from_numpy(np.array(gt_image)).float().to(device) / 255.0
            gt_tensor = gt_tensor.permute(2, 0, 1)
        except Exception as e:
            print(f"无法加载图像 {gt_image_path}: {e}")
            continue

        start_time = time.time()
        rendered_image = render_gaussians(viewpoint_cam, gaussians)
        render_time = time.time() - start_time
        
        # 计算损失
        loss = F.l1_loss(rendered_image, gt_tensor)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        if iteration % 10 == 0:
            scheduler.step()
        
        # 打印进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"迭代 {iteration}/{args.iterations} | 损失: {loss.item():.4f} | "
              f"渲染耗时: {render_time:.2f}s | LR: {current_lr:.6f}")
        
        # 定期保存结果
        if iteration % args.save_interval == 0 or iteration == args.iterations:
            print(f"保存迭代 {iteration} 的检查点...")
            
            # 保存渲染图像
            vis_img = rendered_image.detach().permute(1, 2, 0).cpu().numpy()
            vis_img = np.clip(vis_img * 255, 0, 255).astype(np.uint8)
            Image.fromarray(vis_img).save(os.path.join(args.output_dir, f"render_{iteration:04d}.png"))
            
            # 保存点云状态
            save_point_cloud(
                gaussians, 
                os.path.join(args.output_dir, f"point_cloud_{iteration:04d}.ply")
            )

    print("\n--- 训练完成 ---")
    # 保存最终模型
    torch.save({
        'gaussians': gaussians.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(args.output_dir, "final_model.pth"))

def save_point_cloud(gaussians, file_path):
    """保存当前点云状态为PLY文件"""
    from plyfile import PlyData, PlyElement
    
    # 提取数据
    points = gaussians.means.detach().cpu().numpy()
    colors = (gaussians.colors.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    opacities = torch.sigmoid(gaussians.opacities).detach().cpu().numpy().squeeze()
    
    # 创建顶点数据
    vertex_data = np.zeros(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('opacity', 'f4')
    ])
    
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    vertex_data['opacity'] = opacities
    
    # 创建PLY元素
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # 保存文件
    PlyData([vertex_element]).write(file_path)
    print(f"点云保存至: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("3D Gaussian Splatting Trainer for Apple Silicon")
    parser.add_argument("--source_path", "-s", required=True, type=str, 
                        help="指向由convert.py处理过的COLMAP项目文件夹")
    parser.add_argument("--output_dir", "-o", default="output", type=str,
                        help="输出目录")
    parser.add_argument("--iterations", "-i", type=int, default=100, 
                        help="训练迭代次数")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="保存检查点的频率")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001,
                        help="学习率")
    args = parser.parse_args()
    
    train(args)