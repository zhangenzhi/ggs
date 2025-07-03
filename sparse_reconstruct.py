import pycolmap
import os
from pathlib import Path

# 定义输入和输出路径
# 建议使用绝对路径以避免潜在的相对路径问题
workspace_path = Path('./dataset/3dgs/distorted')
image_path = workspace_path / 'images'
database_path = workspace_path / 'database.db'
sparse_path = workspace_path / 'sparse'

# 确保输出目录存在
os.makedirs(sparse_path, exist_ok=True)

# --- 更正之处 ---
# 使用正确的函数 pycolmap.incremental_mapper
# 并使用正确的参数名 image_dir
summary = pycolmap.incremental_mapping(
    database_path=str(database_path),
    image_path=str(image_path),
    output_path=str(sparse_path)
    # 如果需要，可以在这里传入 options=pycolmap.IncrementalPipelineOptions(...)
)

# 你可以打印重建的摘要信息
if summary:
    print("Reconstruction summary:")
    # summary 是一个字典，包含了重建的统计信息
    for k, v in summary.items():
        print(f"- {k}: {v}")
else:
    print("Reconstruction failed or produced no summary.")

# 重建完成后，生成的稀疏模型会保存在 output_path 目录下
# 你可以像这样加载它来进行后续分析
try:
    reconstruction = pycolmap.Reconstruction(output_path)
    print("\nSuccessfully loaded reconstruction.")
    print(reconstruction.summary())
except ValueError as e:
    print(f"\nCould not load reconstruction: {e}")