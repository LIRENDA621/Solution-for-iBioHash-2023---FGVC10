# -*- coding: utf-8 -*-

import argparse
import os


import pickle

import torch


from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.index import build_index_helper, feature_loader
from pyretri.evaluate import build_evaluate_helper
import pandas as pd
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--config_file', '-cfg', default=None, metavar='FILE', type=str, help='path to config file')
    parser.add_argument('--feat_dir', '-fd', default=None, metavar='FILE', type=str, help='path to feature')
    parser.add_argument('--domain', default='cross_domin', type=str)
    args = parser.parse_args()
    return args

# feat_dir_dict = {
#     "/home/data1/changhao/iBioHash/Results/features/eva_large_336_fr6_e1_full": 1,
#     }

def main():

    # init args
    args = parse_args()
    assert args.config_file is not None, 'a config file must be provided!'

    feat_dir_dict = {
        args.feat_dir: 1,
    }

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.config_file, args.opts)

    # load features
    for feat_dir, weight in feat_dir_dict.items():
        query_fea, gallery_fea, query_names, gallery_names = feature_loader.load(feat_dir)
        
        print(query_fea.shape)

        # build helper and index features
        index_helper = build_index_helper(cfg.index)
        query_fea, query_fea_qe, gallery_fea, dis = index_helper.do_index(query_fea, query_names, gallery_fea)
        
        # 保存输出文件
        output_file_path = os.path.join('/home/data1/changhao/iBioHash/Results/similarity', args.domain, "{}_dba{}_qe{}".format(feat_dir.split('/')[-1], cfg.index.feature_enhancer.DBA.enhance_k, cfg.index.re_ranker.QE.qe_k))

        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        
        with open(os.path.join(output_file_path, 'similarity.pkl'),'wb') as f:
            pickle.dump(dis, f,protocol=4)

        with open(os.path.join(output_file_path,'gallery_dba.pkl'),'wb') as f:
            pickle.dump(gallery_fea.cpu().numpy(), f)

        with open(os.path.join(output_file_path,'query_qe.pkl'),'wb') as f:
            pickle.dump(query_fea_qe.cpu().numpy(), f)

        with open(os.path.join(output_file_path,'query.pkl'),'wb') as f:
            pickle.dump(query_fea.cpu().numpy(), f)

        retreival_results = []
        similarity_index = dis.topk(k=20, dim=1)[1]  # indices
        for i in range(similarity_index.size(0)):
            temp_idx = similarity_index[i].tolist()
            temp_name_list = [gallery_names[j] for j in temp_idx]
            retreival_results.append(' '.join(temp_name_list))
        result_dict = pd.DataFrame({'Id':query_names,'Predicted':retreival_results})
        result_dict[['Id','Predicted']].to_csv(os.path.join(output_file_path, 'submit_feature_posted.csv'), index=False)

        # 生成1-3000
        result_dict_3000 = copy.deepcopy(result_dict)
        for i in range(3000, 10000):
            result_dict_3000.iloc()[i][1] = ''
        result_dict_3000[['Id','Predicted']].to_csv(os.path.join(output_file_path, 'submit_feature_posted_1_3000.csv'), index=False)

        result_dict_6000 = copy.deepcopy(result_dict)
        for i in range(3000):
            result_dict_6000.iloc()[i][1] = ''
        for i in range(6000, 10000):
            result_dict_6000.iloc()[i][1] = ''
        result_dict_6000[['Id','Predicted']].to_csv(os.path.join(output_file_path, 'submit_feature_posted_3001_6000.csv'), index=False)

        result_dict_10000 = copy.deepcopy(result_dict)
        for i in range(6000):
            result_dict_10000.iloc()[i][1] = ''
        result_dict_10000[['Id','Predicted']].to_csv(os.path.join(output_file_path, 'submit_feature_posted_6001_10000.csv'), index=False)

        # 
        result_dict_5000_1 = copy.deepcopy(result_dict)
        for i in range(5000, 10000):
            result_dict_5000_1.iloc()[i][1] = ''
        result_dict_5000_1[['Id','Predicted']].to_csv(os.path.join(output_file_path, 'submit_feature_posted_1_5000.csv'), index=False)

        result_dict_5000_2 = copy.deepcopy(result_dict)
        for i in range(5000):
            result_dict_5000_2.iloc()[i][1] = ''
        result_dict_5000_2[['Id','Predicted']].to_csv(os.path.join(output_file_path, 'submit_feature_posted_5001_10000.csv'), index=False)


        # del index_result_info, query_fea, gallery_fea, dis

        # # build helper and evaluate results
        # evaluate_helper = build_evaluate_helper(cfg.evaluate)
        # mAP, recall_at_k = evaluate_helper.do_eval(index_result_info, gallery_names)

        # # show results
        # evaluate_helper.show_results(mAP, recall_at_k)


if __name__ == '__main__':
    main()

