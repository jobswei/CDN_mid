import numpy as np


# 这里使用随机生成的示例数据
num_actions = 117  # 动作类别数量
num_images = 80  # 图像数量

# 生成随机的预测结果和真实标签（仅作示例）
predicted_labels = np.ones( (num_images,))
true_labels = np.ones((num_images,))

# 创建一个正确性矩阵
correct_mat = np.ones((num_actions,num_images))

# 保存矩阵为npy文件
np.save('/home/wzy/CDN_mod/data/demo_video_fps2/hico/annotations/corre_hico.npy', correct_mat)
