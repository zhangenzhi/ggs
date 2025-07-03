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

def render_gaussians_vectorized(camera, gaussians):
    """向量化渲染器，利用 MPS 并行能力 (已修正和优化)"""
    width, height = int(camera['width']), int(camera['height'])
    fx, fy = camera['fx'], camera['fy']
    cx, cy = camera['cx'], camera['cy']
    
    # 使用正确的相机位姿参数
    R_w2c = camera['R_w2c']
    t_c2w = camera['position']
    
    # 1. 将所有点云从世界坐标系变换到相机坐标系
    means_cam = (gaussians.means - t_c2w) @ R_w2c.T
    
    # 2. 投影到图像平面
    x, y, z = means_cam.unbind(dim=-1)
    
    # 防止z为0或过小导致除零错误
    z_safe = z.clamp(min=1e-6)
    u = (x * fx / z_safe) + cx
    v = (y * fy / z_safe) + cy
    
    # 3. 计算每个高斯的半径
    scales = torch.exp(gaussians.scales)
    avg_scales = torch.mean(scales, dim=1)
    radii = (3 * avg_scales * min(fx, fy) / z_safe).int().clamp(min=1, max=max(width, height))
    
    # 4. 计算不透明度
    opacities = torch.sigmoid(gaussians.opacities).squeeze(-1)
    
    # 5. 过滤无效点
    valid_mask = (z > 0.1) & (u > -radii) & (u < width + radii) & (v > -radii) & (v < height + radii)
    
    if not valid_mask.any():
        return torch.zeros(3, height, width, device=gaussians.device)
    
    # 提取有效点
    u_valid, v_valid, z_valid = u[valid_mask], v[valid_mask], z[valid_mask]
    opacities_valid = opacities[valid_mask]
    colors_valid = gaussians.colors[valid_mask]
    radii_valid = radii[valid_mask]
    
    # 6. 按深度排序 (从远到近)
    sorted_indices = torch.argsort(z_valid, descending=True)
    u_sorted, v_sorted, opacities_sorted, colors_sorted, radii_sorted = \
        u_valid[sorted_indices], v_valid[sorted_indices], opacities_valid[sorted_indices], \
        colors_valid[sorted_indices], radii_valid[sorted_indices]
    
    # 7. 创建图像缓冲区
    image_buffer = torch.zeros(height, width, 4, device=gaussians.device) # 使用4通道 (RGBA)

    # 8. 渲染所有点
    # 这个循环仍然是性能瓶颈，但内部计算已高度优化
    for i in range(len(u_sorted)):
        center_u, center_v = u_sorted[i], v_sorted[i]
        color = colors_sorted[i]
        opacity = opacities_sorted[i]
        radius = radii_sorted[i].item()
        
        # 计算边界框
        min_u, max_u = max(0, int(center_u - radius)), min(width, int(center_u + radius + 1))
        min_v, max_v = max(0, int(center_v - radius)), min(height, int(center_v + radius + 1))
        
        if min_u >= max_u or min_v >= max_v:
            continue
            
        # 创建像素网格
        px = torch.arange(min_u, max_u, device=gaussians.device)
        py = torch.arange(min_v, max_v, device=gaussians.device)
        grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')
        
        # 计算高斯权重
        dx, dy = grid_x - center_u, grid_y - center_v
        dist_sq = dx**2 + dy**2
        sigma_sq = max((radius / 2.0)**2, 1e-6)
        weights = torch.exp(-dist_sq / (2 * sigma_sq))
        
        # Alpha混合
        alpha = opacity * weights
        
        # --- 核心修改：使用 .clone() 避免原地操作 ---
        # 获取当前区域的颜色和alpha值的一个副本
        current_rgba = image_buffer[min_v:max_v, min_u:max_u].clone()
        
        # 计算新的alpha值
        new_alpha = alpha + current_rgba[..., 3] * (1 - alpha)
        # 避免除零
        new_alpha_safe = new_alpha.clamp(min=1e-6)
        
        # 计算新的颜色
        new_color = (color.unsqueeze(0).unsqueeze(0) * alpha.unsqueeze(-1) + 
                     current_rgba[..., :3] * current_rgba[..., 3].unsqueeze(-1) * (1 - alpha.unsqueeze(-1))) / new_alpha_safe.unsqueeze(-1)
        
        # 更新缓冲区
        image_buffer[min_v:max_v, min_u:max_u, :3] = new_color
        image_buffer[min_v:max_v, min_u:max_u, 3] = new_alpha

    # 将return语句移到循环外部
    return image_buffer[..., :3].permute(2, 0, 1)

# --- 3. 主训练逻辑 ---

def train(args):
    print(f"--- 开始训练 (设备: {device}) ---")
    
    # 新增: 启用异常检测以获得更详细的错误信息
    torch.autograd.set_detect_anomaly(True)
    
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
        
        # --- 核心修改：调用向量化渲染器 ---
        rendered_image = render_gaussians_vectorized(viewpoint_cam, gaussians)
        
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
    parser.add_argument("--output_dir", "-o", default="output_vec", type=str,
                        help="输出目录")
    parser.add_argument("--iterations", "-i", type=int, default=100, 
                        help="训练迭代次数")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="保存检查点的频率")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001,
                        help="学习率")
    args = parser.parse_args()
    
    train(args)
    
# python 3dgs_vec.py -s ./dataset/3dgs/distorted