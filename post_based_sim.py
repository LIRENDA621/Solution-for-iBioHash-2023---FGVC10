# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import torch

import math


from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.index import build_index_helper, feature_loader
from pyretri.evaluate import build_evaluate_helper
import pandas as pd
from pyretri.index.metric import KNN
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--config_file', '-cfg', default='/home/data1/changhao/iBioHash/Codes/pytorch-image-models-main/post_config/market_w_tricks.yaml', metavar='FILE', type=str, help='path to config file')
    parser.add_argument('--top_k', type=int, default=None, metavar='top_k')
    parser.add_argument('--post_thr', type=int, default=None, metavar='top_k')
    args = parser.parse_args()
    return args

feat_dir_dict = {
    "/home/data1/changhao/iBioHash/Results/features/eva_e1_enhance": 1,
    }

def main():

    # init args
    args = parse_args()
    assert args.config_file is not None, 'a config file must be provided!'

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.config_file, args.opts)

    similarity=0

    for feat_dir, weight in feat_dir_dict.items():
        # load features
        query_fea, gallery_fea, query_names, gallery_names = feature_loader.load(feat_dir)
        #---------------
        query_fea, gallery_fea = torch.Tensor(query_fea), torch.Tensor(gallery_fea)

        if torch.cuda.is_available():
            query_fea = query_fea.cuda()
            gallery_fea = gallery_fea.cuda()
            print("loading to the GPU")
        print(query_fea.shape)

        metric = KNN()
        dis, sorted_index = metric(query_fea, query_fea) # + 2

        # ======== query 分组 ==============
        mode_split = 'method2_10'
        if mode_split == 'method2_10':
            skip_i = []
            group_select = [[] for i in range(11)]
            # 寻找公共子集
            for topk_select in range(10, 5, -1):   
                group_topk_select = []
                print('topk: ' +str(topk_select))
                for i in range(len(sorted_index)):
                    if i in skip_i:
                        continue
        
                    topk_index = sorted_index[i][:topk_select]
                    
                    temp_index = []
                    for top_i in range(len(topk_index) - 1):
                        temp_index.append(sorted_index[topk_index[top_i + 1]][:topk_select].sort().values)

                    elements = dict()  
                    temp_index.append(topk_index.sort().values)
                    for ti in range(len(temp_index)):
                        for q_temp in temp_index[ti]:
                            q_temp = int(q_temp)
                            elements[q_temp] = elements.get(q_temp, 0) + 1

                    public_subset = []
                    for el in elements: 
                        if elements[el] >= topk_select - 2:
                            public_subset.append(el)
                    
                    if len(public_subset) < topk_select - 2: 
                        continue
                    
                    flag_re = False
                    for s_index in public_subset:
                        if s_index in skip_i:
                            flag_re = True
                            break
                    if flag_re:
                        # print("re condition")
                        continue  
                    group_topk_select.append([int(s_index) for s_index in public_subset])
                    skip_i.extend([int(s_index) for s_index in public_subset]) 

                for gts in range(len(group_topk_select)):
                    length_gts = len(group_topk_select[gts])
                    group_select[length_gts].append(group_topk_select[gts])

            group_3 = []
            for i in range(len(sorted_index)):
                if i in skip_i:
                    continue
                top3_index = sorted_index[i][:3]
                temp1_index = sorted_index[top3_index[1]][:3].sort().values
                temp2_index = sorted_index[top3_index[2]][:3].sort().values

                if top3_index.equal(temp1_index) and top3_index.equal(temp2_index):
                    flag_re = False  
                    if int(top3_index[1]) in skip_i or int(top3_index[2]) in skip_i:
                        flag_re = True
                        continue  
                    skip_i.extend([i, int(top3_index[1]), int(top3_index[2])])                      
                    group_3.append([i, int(top3_index[1]), int(top3_index[2])])
            group_select[3] = group_3

            group_2 = []
            for i in range(len(sorted_index)):
                if i in skip_i:
                    continue
                top2_index = sorted_index[i][:2]
                temp1_index = sorted_index[top2_index[1]][:2].sort().values

                if top2_index.equal(temp1_index):
                    flag_re = False  
                    if int(top2_index[1]) in skip_i:
                        flag_re = True
                        continue
                    skip_i.extend([i, int(top2_index[1])]) 
                    group_2.append([i, int(top2_index[1])])
            group_select[2] = group_2

            stat_q_num = 0
            stat_g_num = 0
            for i in range(2, len(group_select)):
                stat_q_num += (i * len(group_select[i]))
                stat_g_num += (20 * len(group_select[i]))
    

            group_select.pop(0)
            group_select.pop(0)


        # ======= gallery 分配 =======
        with open(os.path.join('/home/data1/changhao/iBioHash/Results/fused_results/cross_beit_e1_eva_e1/similarity.pkl'), "rb") as f:
            qg_dis = pickle.load(f)
        qg_sorted_index = []
        for i in range(math.ceil(qg_dis.size(0)/400)):
            qg_sorted_index.append(torch.argsort(qg_dis[i*400: min((i+1)*400, qg_dis.size(0))], dim=1, descending=True))
        qg_sorted_index = torch.cat(qg_sorted_index, dim=0)

        qg_sorted_index_assign = torch.zeros((10000, 20), dtype=int)
        used_gallery_index = []
        
        for group_idx in range(9): 
            group_topk_select = group_select[group_idx]
            for group_i in range(len(group_topk_select)):
                used_gallery_index_pergroup = []  
                current_gallery_point = [0] * len(group_topk_select[group_i])

                while len(used_gallery_index_pergroup) != 20:
                    for q_index_i in range(len(group_topk_select[group_i])):
                        cur_q_index = group_topk_select[group_i][q_index_i]
                        cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]])  # 当前gallery id
                        
                        while cur_g_index in used_gallery_index:  
                            current_gallery_point[q_index_i] += 1
                            cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]])
                        
                        used_gallery_index_pergroup.append(cur_g_index)
                        used_gallery_index.append(cur_g_index)

                        if len(used_gallery_index_pergroup) == 20:
                            break

                g_sorted_index_pergroup = torch.tensor(used_gallery_index_pergroup)
                for q_index_i in range(len(group_topk_select[group_i])):
                    cur_q_index = group_topk_select[group_i][q_index_i]
                    qg_sorted_index_assign[cur_q_index] = g_sorted_index_pergroup

        re_assign_single_q = {}
        sim_cnt_thr = args.top_k
        for i in range(10000):
            if i in skip_i:
                continue
            
            sim_among_group = [[] for i in range(9)] 
            for group_topk_select_i in range(len(group_select)):
                group_topk_select = group_select[group_topk_select_i]

                for group_i in range(len(group_topk_select)):
                    q_idex = group_topk_select[group_i][0]
                    gallery_index_set = set(qg_sorted_index_assign[q_idex].numpy())
                    cur_q_sim_results_set = set(qg_sorted_index[i][:args.post_thr].numpy())
                    sim_cnt = len(gallery_index_set.intersection(cur_q_sim_results_set))
                    
                    sim_among_group[group_topk_select_i].append(sim_cnt)


            max_sim, max_i, max_j = -float('inf'), 0, 0
            for sim_i in range(len(sim_among_group)):
                for sim_j in range(len(sim_among_group[sim_i])):
                    if sim_among_group[sim_i][sim_j] > max_sim:
                        max_sim = sim_among_group[sim_i][sim_j]
                        max_i = sim_i
                        max_j = sim_j
            if max_sim >= sim_cnt_thr:
                re_assign_single_q[i] = (max_i, max_j)
        
        for single_q in re_assign_single_q:
            re_assign_i, re_assign_j = re_assign_single_q[single_q]
            group_select[re_assign_i][re_assign_j].append(single_q)
            qg_sorted_index_assign[single_q] = qg_sorted_index_assign[group_select[re_assign_i][re_assign_j][0]]
            skip_i.append(single_q)

       
        # 将单个的query分配给 已分好的组


        group_single_select = []
        for i in range(10000):
            if i in skip_i:
                continue
            group_single_select.append(i)
        
        used_gallery_index_pergroup = [] 
        current_gallery_point = [0] * len(group_single_select)

        for topk_i in range(20):
            for q_index_i in range(len(group_single_select)):
                cur_q_index = group_single_select[q_index_i]  

                if cur_q_index in skip_i:
                    continue

                cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]]) 
                while cur_g_index in used_gallery_index:  
                    current_gallery_point[q_index_i] += 1  
                    if current_gallery_point[q_index_i] >= 270: 
                        skip_i.append(cur_q_index)
                        break
                    cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]]) 
                
                if cur_q_index in skip_i:
                    continue
                
                qg_sorted_index_assign[cur_q_index][topk_i] = cur_g_index
                used_gallery_index.append(cur_g_index)

        print("assign gallery done")

        retreival_results = []
        for i in range(qg_sorted_index_assign.size(0)):
            temp_idx = qg_sorted_index_assign[i].tolist()
            temp_name_list = [gallery_names[j] for j in temp_idx]
            retreival_results.append(' '.join(temp_name_list))
        

        result_dict = pd.DataFrame({'Id':query_names,'Predicted':retreival_results})
        result_dict[['Id','Predicted']].to_csv(os.path.join('/home/data1/changhao/iBioHash/Codes/pytorch-image-models-main/lrd_post_sim_submit', 'submit_post_based_sim_method21_eva_beit.csv'), index=False)
    


if __name__ == '__main__':
    main()

