import numpy as np
from plyfile import PlyData, PlyElement


file_path = "Yingrenshi_whole_scene.txt"

def txt_to_ply(txt_file, ply_file, wo_sem=False):
    # 读取txt文件内容
    data = np.loadtxt(txt_file, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                       ('semantic_label', 'i4'), ('instance_label', 'i4'),
                                       ('fine_grained_category', 'i4')])

    # 构造点云数据，PLY文件只需要x, y, z, red, green, blue
    ply_data = np.array([(row['x'], row['y'], row['z'], row['red'], row['green'], row['blue']) 
                         for row in data], 
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # 创建PLY元素
    ply_el = PlyElement.describe(ply_data, 'vertex')

    # 写入PLY文件
    PlyData([ply_el], text=True).write(ply_file)

    print(f"Number of points  downsampling: {len(ply_data.shape[0])}")

# 示例调用
txt_to_ply(file_path, 'output.ply')
