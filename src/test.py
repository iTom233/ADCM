# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap.umap_ as umap
# from mpl_toolkits.mplot3d import Axes3D

# # 加载pkl数据
# with open('./gym/data/hopper-medium-replay-v2.pkl', 'rb') as f:
#     data = pickle.load(f)

# # 提取状态和动作
# n = len(data)
# observations = []
# actions = []
# for data_i in data:
#     for i in range(len(data_i['observations'])):
#         observations.append(data_i['observations'][i])  # 形状 (num_samples, 11)
#         actions.append(data_i['actions'][i])            # 形状 (num_samples, 3)
# observations = np.array(observations)
# actions = np.array(actions)
# state_action = np.concatenate([observations, actions], axis=1)
# pca = PCA(n_components=3)
# sa_pca_3d  = pca.fit_transform(state_action)

# def plot_3d_scatter(embedding, title):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
#                s=1, alpha=0.5, c=embedding[:, 2], cmap='viridis')
#     ax.set_title(title)
#     ax.set_xlabel('Component 1')
#     ax.set_ylabel('Component 2')
#     ax.set_zlabel('Component 3')
#     plt.show()
# plot_3d_scatter(sa_pca_3d, '3D PCA Projection')

import gym
import d4rl

env = gym.make('door-cloned-v1')
obs = env.reset()
print("环境初始化成功！")
# from gym import envs
# print(envs.registry.all())