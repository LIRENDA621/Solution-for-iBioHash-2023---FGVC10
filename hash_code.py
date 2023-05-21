import csv
import os
import hashlib

cvs_file=''
path=''
submit_hash=''
with open('cvs_file', 'r') as file:
    reader = csv.DictReader(file)
    
    predictions_dict = {}
    
    for row in reader:
        id_val = row['Id']
        predicted_val = row['Predicted']
        

        predicted_images = predicted_val.split()
        
        predicted_images = [image + '.jpg' for image in predicted_images]
        
        if predicted_val in predictions_dict:
            predictions_dict[predicted_val]['Id'].append(id_val)
            predictions_dict[predicted_val]['Predicted'].extend(predicted_images)
        else:
            predictions_dict[predicted_val] = {'Id': [id_val], 'Predicted': predicted_images}
    
    predictions_list = [[*v['Predicted'], *v['Id']] for k, v in predictions_dict.items()]
hash_dict={}
for sub_list in predictions_list:
    code=None
    for image in sub_list:
        image=os.path.join(path,image)
        if os.path.exists(image):
            with open(image,'rb') as f:
                img_data=f.read() 
                m=hashlib.md5()
                m.update(img_data)
                code=m.hexdigest()[:12]
                code=bin(int(code,16))[2:].zfill(48)
        if code is not None:
            break
    for image in sub_list:
        hash_dict[image]=code
    
for image,code in hash_dict.items():
    csv_data=[image,code]
    with open(submit_hash,'a',newline='') as f:
        writer=csv.writer(f)
        writer.writerow((csv_data))
        
        