from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import shutil
import os
import pickle
import torch
from tqdm import tqdm
import math
os.environ['CUDA_VISIBLE_DEVICES']='1'
num_clusters=2750
#/home/data1/changhao/iBioHash/Results/features/beit_large_query_gallery_fr12_ckp/ckp_2
with open('/home/data1/changhao/iBioHash/Results/features/beit_large_query_gallery_fr12_ckp/ckp_24/gallery.pkl', 'rb') as f:
    # Load object from file
    gallery = pickle.load(f)
    
with open('/home/data1/changhao/iBioHash/Results/features/beit_large_query_gallery_fr12_ckp/ckp_24/query.pkl', 'rb') as f:
    # Load object from file
    query = pickle.load(f)

with open('/home/data1/changhao/iBioHash/Results/features/eva_large_336_fr6_e1_full/names.pkl', 'rb') as f:
    # Load object from file
    names = pickle.load(f)

print('load data')

gallery_fea = torch.tensor(gallery)
query_fea = torch.tensor(query)
all_fea = torch.concatenate((query_fea, gallery_fea), axis=0).cuda()

sim_all = torch.zeros((all_fea.size(0), all_fea.size(0)))
for i in tqdm(range(math.ceil(all_fea.size(0)/400))):
    sim_all[i*400: min((i+1)*400, all_fea.size(0))] = all_fea[i*400: min((i+1)*400, all_fea.size(0))].mm(all_fea.t()) + 2

sim_all = sim_all.numpy().astype(np.float32)
clustering = SpectralClustering(n_clusters=4000, affinity = 'precomputed', assign_labels='discretize')
clustering.fit(sim_all)
# clustering = SpectralClustering(n_clusters=4000, affinity = 'nearest_neighbors', n_neighbors = 20, assign_labels='discretize')
# clustering.fit(all_fea.cpu().numpy())



output_folder='/home/data1/zgp/spectral_kmeans_4000_20_qg_beit_24ep'
query_samples = names["query"].samples
gallery_samples = names["gallery"].samples
query_image_names = [x[0] for x in query_samples]
gallery_image_names = [x[0] for x in gallery_samples]
query_image_names.extend(gallery_image_names)
# 将所有特征向量合并到一个大数组中
all_features = np.concatenate((query, gallery), axis=0)
# similarity_matrix = cosine_similarity(all_features)
# similarity_matrix = np.nan_to_num(similarity_matrix,nan=0.0, posinf=1.0, neginf=0.0)
# 调用SpectralClustering算法进行聚类
clustering = SpectralClustering(n_clusters=4000, n_neighbors=20, assign_labels='discretize')

# 将特征传递给算法进行聚类
clustering.fit(all_features)
# 获取聚类标签
labels = clustering.labels_
print(labels)

# 将每个图片文件分配到它所属的聚类，并保存到相应的文件夹中
for i, file_name in enumerate(query_image_names):
    cluster_label = labels[i]
    label_folder = os.path.join(output_folder, f"cluster_{cluster_label}")
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    file_name2 = os.path.basename(file_name)
    label_name=label_folder+'/'+file_name2
    shutil.copyfile(file_name,label_name)