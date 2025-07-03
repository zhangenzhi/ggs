import pycolmap
import os

# 定义输入和输出路径
# 建议使用绝对路径以避免潜在的相对路径问题
base_path = './dataset/3dgs/distorted'
database_path = os.path.join(base_path, 'database.db')
image_dir = os.path.join(base_path, 'images')  # 注意：参数名是 image_dir
output_path = os.path.join(base_path, 'sparse')

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# --- 更正之处 ---
# 使用正确的函数 pycolmap.incremental_mapper
# 并使用正确的参数名 image_dir
summary = pycolmap.incremental_mapping(
    database_path=database_path,
    image_dir=image_dir,
    output_path=output_path,
    # 你还可以通过 options 字典传递更详细的 COLMAP 参数
    # 例如: options={'min_num_matches': 30}
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