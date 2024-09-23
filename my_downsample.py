import numpy as np
import open3d as o3d
import os
from plyfile import PlyData, PlyElement

def storePly(path, xyz, rgb, wo_normal=False, text=False):
    # Define the dtype for the structured array

    if wo_normal:
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        attributes = np.concatenate((xyz, rgb), axis=1)
    else:
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        normals = np.zeros_like(xyz)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element], text=text)
    ply_data.write(path)

path = 'output.ply'
voxel_size = 0.1

source = o3d.io.read_point_cloud(path)
source_down = source.voxel_down_sample(voxel_size)
output_path = '/'.join(path.split('/')[:-1] + ['points3D_down.ply'])
# o3d.io.write_point_cloud(output_path, source_down)

# 从source_down中提取xyz坐标和rgb颜色信息
xyz = np.asarray(source_down.points)

# 检查是否有颜色信息
if source_down.has_colors():
    rgb = (np.asarray(source_down.colors) * 255).astype(np.uint8)
else:
    rgb = np.zeros_like(xyz, dtype=np.uint8)  # 如果没有颜色，则设置为默认值

# 使用storePly保存点云
storePly(output_path, xyz, rgb, wo_normal=True, text=True)

print(f"Number of points  downsampling: {len(source.points)} -> {len(source_down.points)}")
