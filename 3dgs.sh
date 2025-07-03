export KMP_DUPLICATE_LIB_OK=TRUE
python 3dgs.py -s ./dataset/3dgs/distorted

colmap mapper \
    --database_path ./dataset/3dgs/distorted/database.db \
    --image_path ./dataset/3dgs/distorted/images \
    --output_path ./dataset/3dgs/distorted/sparse \