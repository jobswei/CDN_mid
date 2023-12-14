import scipy.io
import numpy as np
import pandas as pd

# 读取MAT文件
mat_data = scipy.io.loadmat("/home/wzy/CDN/data/hico_20160224_det/anno_bbox.mat")

# 获取MAT文件中的变量
mat_variable = mat_data['list_action']  # 替换为实际的变量名

# 将MAT变量转换为NumPy数组
numpy_array = np.array(mat_variable)

# 将NumPy数组转换为Pandas DataFrame（可选）
dataframe = pd.DataFrame(numpy_array)

# 保存DataFrame为CSV文件
dataframe.to_csv('output_file.csv', index=False)  # 将数据保存为CSV文件，index=False表示不保存行索引
