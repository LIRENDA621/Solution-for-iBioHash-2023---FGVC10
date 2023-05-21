# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import torch


from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.index import build_index_helper, feature_loader
from pyretri.evaluate import build_evaluate_helper
import pandas as pd
from pyretri.index.metric import KNN
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, kmeans_plusplus

import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 跨域
feat_dir_dict = {
    "/home/data1/changhao/iBioHash/Results/similarity/eva_best_e1": 0.1,
    "/home/data1/changhao/iBioHash/Results/similarity/beit_best_e1": 0.9,
}

output_file_path = os.path.join('/home/data1/changhao/iBioHash/Results/fused_results', 'cross_beit_e1_eva_e1') 
# ===================================================
with open(os.path.join('/home/data1/changhao/iBioHash/Results/features/eva_best_e1_1','names.pkl'), "rb") as f:
    data = pickle.load(f)
    query_names = [item[0].name.split("/")[-1] for item in data['query']]
    gallery_names = [item[0].name.split("/")[-1].split('.')[0] for item in data['gallery']]

if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)


fuse_mode = 'sum'
if fuse_mode == 'sum':
    similarity = 0
    for fea_dir, weight in feat_dir_dict.items():

        with open(os.path.join(fea_dir,'similarity.pkl'), "rb") as f:
            sim = pickle.load(f)
        

        similarity += sim * weight

    with open(os.path.join(output_file_path, 'similarity.pkl'),'wb') as f:
        pickle.dump(similarity, f,protocol=4)

    retreival_results = []
    similarity_index = similarity.topk(k=20, dim=1)[1]  # indices
    for i in range(similarity_index.size(0)):
        temp_idx = similarity_index[i].tolist()
        temp_name_list = [gallery_names[j] for j in temp_idx]
        retreival_results.append(' '.join(temp_name_list))
    result_dict = pd.DataFrame({'Id':query_names,'Predicted':retreival_results})

    submit_file_name = 'submit_fused_{}'.format(output_file_path.split('/')[-1])

elif fuse_mode == 'concate':
    
    similarity_index_all = []
    retreival_results = []

    for fea_dir, weight in feat_dir_dict.items():
        with open(os.path.join(fea_dir,'similarity.pkl'), "rb") as f:
            sim = pickle.load(f)
        similarity_index = sim.topk(k=20, dim=1)[1]
        similarity_index_all.append(similarity_index)

    for query_index in range(10000): 
        temp_fused_topk_list = []

        current_point = [0 for _ in range(len(similarity_index_all))]  
        while len(temp_fused_topk_list) != 20:
            for model_i in range(len(similarity_index_all)): 
                cur_gallery_index = int(similarity_index_all[model_i][query_index][current_point[model_i]])
                if cur_gallery_index not in temp_fused_topk_list:
                    temp_fused_topk_list.append(cur_gallery_index)
                
                if len(temp_fused_topk_list) == 20:
                    break
                current_point[model_i] += 1

        temp_name_list = [gallery_names[j] for j in temp_fused_topk_list]
        retreival_results.append(' '.join(temp_name_list))
    result_dict = pd.DataFrame({'Id':query_names,'Predicted':retreival_results})
    submit_file_name = 'submit'

result_dict[['Id','Predicted']].to_csv(os.path.join(output_file_path, '{}.csv'.format(submit_file_name)), index=False)

result_dict_3000 = copy.deepcopy(result_dict)
for i in range(3000, 10000):
    result_dict_3000.iloc()[i][1] = ''
result_dict_3000[['Id','Predicted']].to_csv(os.path.join(output_file_path, '{}_1_3000.csv'.format(submit_file_name)), index=False)

result_dict_6000 = copy.deepcopy(result_dict)
for i in range(3000):
    result_dict_6000.iloc()[i][1] = ''
for i in range(6000, 10000):
    result_dict_6000.iloc()[i][1] = ''
result_dict_6000[['Id','Predicted']].to_csv(os.path.join(output_file_path, '{}_3001_6000.csv'.format(submit_file_name)), index=False)

result_dict_10000 = copy.deepcopy(result_dict)
for i in range(6000):
    result_dict_10000.iloc()[i][1] = ''
result_dict_10000[['Id','Predicted']].to_csv(os.path.join(output_file_path, '{}_6001_10000.csv'.format(submit_file_name)), index=False)

result_dict_5000_1 = copy.deepcopy(result_dict)
for i in range(5000, 10000):
    result_dict_5000_1.iloc()[i][1] = ''
result_dict_5000_1[['Id','Predicted']].to_csv(os.path.join(output_file_path, '{}_1_5000.csv'.format(submit_file_name)), index=False)

result_dict_5000_2 = copy.deepcopy(result_dict)
for i in range(5000):
    result_dict_5000_2.iloc()[i][1] = ''
result_dict_5000_2[['Id','Predicted']].to_csv(os.path.join(output_file_path, '{}_5001_10000.csv'.format(submit_file_name)), index=False)


        

print("-")
