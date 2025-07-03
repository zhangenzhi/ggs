import pycolmap
import os

# 定义输入和输出路径
# 建议使用绝对路径以避免潜在的相对路径问题
base_path = './dataset/3dgs/distorted'
database_path = os.path.join(base_path, 'database.db')
image_path = os.path.join(base_path, 'images')
output_path = os.path.join(base_path, 'sparse')

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 调用 pycolmap.mapper 函数
# 该函数会运行增量式重建，并返回一个字典，包含了重建的摘要信息
summary = pycolmap.mapper(
    database_path=database_path,
    image_path=image_path,
    output_path=output_path
)

# 你可以打印重建的摘要信息
if summary:
    print("Reconstruction summary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")
else:
    print("Reconstruction failed or produced no summary.")

# 重建完成后，你可以选择性地加载重建结果进行分析
a = pycolmap.Reconstruction(output_path)
print(a.summary())