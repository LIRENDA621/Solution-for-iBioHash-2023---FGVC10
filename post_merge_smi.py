
import csv

def merge_lists(csv_file):
    id_predicted_map = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            id_value = row[0]
            predicted_value = row[1]

            if predicted_value not in id_predicted_map:
                id_predicted_map[predicted_value] = []

            id_predicted_map[predicted_value].append(id_value)

    merged_list = []
    for predicted_value, id_list in id_predicted_map.items():
        id_list.extend(predicted_value.split())
        merged_list.append(id_list)

    return merged_list

import os
import hashlib
csv_file = '/home/data1/changhao/iBioHash/submit_post_based_sim_method21_eva_beit_cross_domain_thr5_top40.csv'
query_path='/home/data1/changhao/iBioHash/Datasets/iBioHash_Query/Query'
result = merge_lists(csv_file)
hash_dict={}
for sublist in result:
    for img in sublist:
        img_name=os.path.join(query_path, img)
        if os.path.exists(img_name):
            break
    code=None
    with open(img_name, 'rb') as f:
            img_data = f.read()
            m = hashlib.md5()
            m.update(img_data)
            code = m.hexdigest()[:12]
            code = bin(int(code, 16))[2:].zfill(48)
    for img in sublist:
        img_id = img
        hash_dict[img_id] = code
        
    if code is not None:
        code = hashlib.md5(code.encode('utf-8')).hexdigest()[:12]
    
    # 遍历完成一次子文件夹，将结果写入csv
    for img_id, code in hash_dict.items():
        csv_data = [img_id, code]
        with open('final_hash.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_data)
    # 重置hash_dict
    hash_dict = {}



