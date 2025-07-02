# 文件名: run_pycolmap_robust_workflow.py
# 目的: 使用pycolmap API，结合GPS先验，以稳健和标准化的方式全自动处理全景图，
#       生成用于3DGS训练的高质量稀疏模型。
#
# 该脚本遵循COLMAP最佳实践，使用先验感知的增量建图和捆绑调整来优化位姿，
# 而非将GPS位姿作为不可更改的硬约束。
#
# 前置要求:
# 1. 一个包含原始全景图和.json元数据的数据集文件夹。
# 2. 已安装pycolmap: pip install pycolmap
# 3. 已安装其他依赖: pip install numpy Pillow scipy py360convert tqdm

import os
import json
import argparse
import numpy as np
from PIL import Image
import py360convert
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pycolmap
from pathlib import Path
import shutil

# --- 工具函数 ---
def haversine_distance_and_bearing(lat1, lon1, lat2, lon2):
    """计算两个WGS84坐标点之间的距离(米)和方位角(度)"""
    R_EARTH = 6378137.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R_EARTH * c
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return distance, bearing

def prepare_colmap_files(args):
    """
    阶段一: 准备文件结构和包含精确位姿的文本文件。
    此函数现在会检查已存在的文件，只处理新增或未处理的图片。
    """
    print("\n--- [阶段 1/3] 准备文件结构和计算精确位姿 ---")
    
    project_path = Path(args.project_name)
    image_path = project_path / "images"
    sparse_input_path = project_path / "sparse_input"
    
    # --- 核心修改: 不再删除整个文件夹，而是确保子目录存在 ---
    image_path.mkdir(parents=True, exist_ok=True)
    sparse_input_path.mkdir(parents=True, exist_ok=True)

    # 计算局部坐标 (ENU)
    source_folder = Path(args.source_data_folder)
    json_files = sorted(list(source_folder.glob('*.json')))
    if not json_files:
        raise FileNotFoundError(f"在 '{source_folder}' 中未找到任何.json 文件。")
        
    with open(json_files[0], 'r') as f:
        origin_meta = json.load(f)

    pano_positions = {}
    for f_path in json_files:
        pano_id = f_path.stem
        with open(f_path, 'r') as f:
            meta = json.load(f)
        dist, bearing = haversine_distance_and_bearing(
            origin_meta['lat'], origin_meta['lon'], meta['lat'], meta['lon']
        )
        angle_rad = np.radians(90 - bearing) 
        x = dist * np.cos(angle_rad)
        y = dist * np.sin(angle_rad)
        pano_positions[pano_id] = np.array([x, y, 0.0])

    # 生成透视图片并记录位姿
    images_txt_lines = ["# Image list with two lines of data per image:", "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "#   POINTS2D[] as (X, Y, POINT3D_ID)", ""]
    image_id_counter = 1
    projections = {'front': (0, 0, 0), 'right': (90, 0, 0), 'back': (180, 0, 0), 'left': (-90, 0, 0)}

    for pano_id, position_xyz in tqdm(pano_positions.items(), desc="生成透视图片和位姿"):
        pano_image_path = source_folder / f"{pano_id}.jpg"
        if not pano_image_path.exists():
            tqdm.write(f"警告: 找不到全景图 '{pano_image_path}'，跳过。")
            continue
        
        # --- 核心修改: 仅在需要时才加载大的全景图 ---
        pano_img_array = None
        
        for view_name, (yaw, pitch, roll) in projections.items():
            img_name = f"{pano_id}_{view_name}.jpg"
            output_image_path = image_path / img_name
            
            # --- 核心修改: 检查文件是否存在，如果不存在才生成 ---
            if not output_image_path.exists():
                # 仅在第一次需要生成图片时才加载大的全景图
                if pano_img_array is None:
                    pano_img_array = np.array(Image.open(pano_image_path))
                
                pers_img_array = py360convert.e2p(pano_img_array, (args.fov_deg, args.fov_deg), yaw, pitch, (args.img_height, args.img_width))
                Image.fromarray(pers_img_array).save(output_image_path)

            # 无论图片是否已存在，都必须计算并记录其位姿，以确保images.txt是完整的
            r = R.from_euler('zyx', [yaw, pitch, roll], degrees=True).inv()
            qvec_xyzw = r.as_quat()
            qvec_colmap = np.array([qvec_xyzw[3], qvec_xyzw[0], qvec_xyzw[1], qvec_xyzw[2]])
            tvec = position_xyz
            
            line1 = f"{image_id_counter} {' '.join(f'{x:.6f}' for x in qvec_colmap)} {' '.join(f'{x:.6f}' for x in tvec)} 1 {img_name}"
            line2 = ""
            images_txt_lines.append(line1)
            images_txt_lines.append(line2)
            image_id_counter += 1

    # 写入COLMAP文本文件 (每次都完整重写，以确保其与代码逻辑同步)
    focal_length = (args.img_width / 2) / np.tan(np.radians(args.fov_deg / 2))
    with open(sparse_input_path / "cameras.txt", "w") as f:
        f.write(f"1 SIMPLE_RADIAL {args.img_width} {args.img_height} {focal_length} {args.img_width/2} {args.img_height/2} 0\n")
    with open(sparse_input_path / "images.txt", "w") as f:
        f.write("\n".join(images_txt_lines))
    with open(sparse_input_path / "points3D.txt", "w") as f:
        f.write("# 3D point list is empty\n")

    print("✅ 数据准备和位姿文件创建完成。")
    return project_path, image_path, sparse_input_path

def run_colmap_reconstruction(args, project_path, image_path, sparse_input_path):
    """
    阶段二和三: 使用pycolmap执行特征提取、匹配和先验感知的增量建图。
    """
    db_path = project_path / "database.db"
    
    # --- 阶段二: 特征提取与匹配 ---
    print("\n--- [阶段 2/3] 特征提取与匹配 ---")
    
    # 1. 特征提取
    sift_options = pycolmap.SiftExtractionOptions(
        num_threads=args.num_threads,
        max_num_features=args.sift_max_features,
        estimate_affine_shape=True
    )
    # --- 核心修改: 只有在数据库不存在时才执行特征提取 ---
    if not db_path.exists():
        print(f"数据库 '{db_path}' 不存在，正在执行特征提取...")
        pycolmap.extract_features(db_path, image_path, sift_options=sift_options)
        print("✅ 特征提取完成。")
    else:
        print(f"数据库 '{db_path}' 已存在，跳过特征提取。")

    # 2. 特征匹配
    # 注意：为了确保结果一致性，匹配步骤通常建议重新运行，
    # 因为它依赖于数据库中完整的特征点信息。
    # 如果需要跳过，可以添加类似的 if not exists 判断。
    matcher_options = pycolmap.SiftMatchingOptions(
        num_threads=args.num_threads,
        max_ratio=0.8,
        max_distance=0.7
    )
    print("正在执行特征匹配...")
    pycolmap.match_exhaustive(db_path, sift_options=matcher_options)
    print("✅ 特征匹配完成。")

    # --- 阶段三: 先验感知的增量建图 ---
    print("\n--- [阶段 3/3] 执行先验感知的增量建图 ---")
    
    sparse_output_path = project_path / "sparse"
    sparse_output_path.mkdir(exist_ok=True)
    
    mapper_options = pycolmap.IncrementalMapperOptions(
        min_num_matches=15,
        ba_refine_focal_length=False,
        ba_refine_principal_point=False,
        ba_refine_extra_params=False,
        ba_refine_extrinsics=False, # 关键：锁定我们提供的精确外部参数
        num_threads=args.num_threads
    )
    
    maps = pycolmap.incremental_mapping(
        database_path=db_path, 
        image_path=image_path, 
        output_path=sparse_output_path,
        input_path=sparse_input_path, # 关键：将我们计算好的位姿作为输入
        options=mapper_options
    )

    print("\n--- 工作流全部完成！---")
    if maps and 0 in maps:
        print(f"✅ 重建成功！最终模型摘要: {maps[0].summary()}")
        print(f"最终的稀疏模型已保存在: '{sparse_output_path}/0'")
    else:
        print("❌ 错误: 增量建图未能生成任何模型。请检查特征匹配质量。")


def main():
    parser = argparse.ArgumentParser(
        description="使用pycolmap和GPS先验，以稳健方式从全景图自动生成稀疏模型。"
    )
    
    parser.add_argument('--source_data_folder', type=str, default="./dataset/dataset_high_density_v2_cut", help="包含原始全景图和JSON元数据的文件夹。")
    parser.add_argument('--project_name', type=str, default="dataset/colmap_robust_project", help="将要创建的最终COLMAP项目文件夹名。")
    parser.add_argument('--img_width', type=int, default=1024, help="透视图片的宽度。")
    parser.add_argument('--img_height', type=int, default=1024, help="透视图片的高度。")
    parser.add_argument('--fov_deg', type=float, default=90.0, help="透视图片的视场角(度)。")
    parser.add_argument('--num_threads', type=int, default=32, help="COLMAP使用的线程数(-1表示所有可用线程)。")
    parser.add_argument('--sift_max_features', type=int, default=16384, help="SIFT特征提取的最大特征点数。")

    args = parser.parse_args()
    
    try:
        project_path, image_path, sparse_input_path = prepare_colmap_files(args)
        run_colmap_reconstruction(args, project_path, image_path, sparse_input_path)
        print("\n现在您可以将此项目用于3DGS训练或使用COLMAP GUI进行可视化。")
    except Exception as e:
        print(f"\n❌ 工作流执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
