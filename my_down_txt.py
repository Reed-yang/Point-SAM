import numpy as np

# 定义文件路径
file_path = './Yingrenshi_whole_scene.txt'
building_rate = 10
non_building_rate = 100

# 定义数据类型结构
dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
         ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
         ('semantic_label', 'i4'), ('instance_label', 'i4'),
         ('fine_grained_category', 'i4')]

# 从文件中加载数据
data = np.loadtxt(file_path, dtype=dtype)

# 分离出 Semantic_label 为 6 的 "Building" 数据
building_data = data[data['semantic_label'] == 6]

sample_size = building_data.shape[0] // building_rate
downsampled_building_data = np.random.choice(building_data, sample_size, replace=False)
print(f"Building points downsampling: {len(building_data)} -> {len(downsampled_building_data)}")

# 分离出非 "Building" 数据
non_building_data = data[data['semantic_label'] != 6]

# 对非 "Building" 的数据进行下采样（10倍）
# 使用 numpy 的 np.random.choice 进行随机采样
sample_size = non_building_data.shape[0] // non_building_rate
downsampled_non_building_data = np.random.choice(non_building_data, sample_size, replace=False)
print(f"Non-Building points downsampling: {len(non_building_data)} -> {len(downsampled_non_building_data)}")


# 合并 "Building" 数据和下采样后的非 "Building" 数据
final_data = np.concatenate([downsampled_building_data, downsampled_non_building_data])
print(f"Final points downsampling: {len(data)} -> {len(final_data)}, down rate {len(final_data)/len(data)}")

# 保存结果到新的文件
output_path = './yingrenshi_downsampled.txt'
np.savetxt(output_path, final_data, fmt='%.4f %.4f %.4f %d %d %d %d %d %d')

print(f'降采样后的数据已保存到: {output_path}')
