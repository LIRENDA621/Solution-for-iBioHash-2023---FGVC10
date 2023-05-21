# Top 1 solution for iBioHash_2023_competition
Top 1 (1/23)  solution for iBioHash_2023_competition

Thanks to my team members!

### Requirements
* Python 3.8
* cuda 11.7
* timm 0.8.17
* pytorch 1.13.1
* torchaudio 0.13.1
* torchvision 0.14.1

### About the Code

#### 1. Prepare Data
Download the competition data according to (https://www.kaggle.com/competitions/ibiohash-2023-fgvc10/data)

Organize data into ImageNet dataset format
#### 2. Train the Model
```
python train.py [--ages_name args]
```
#### 3. Feature Extraction
Here the feature extraction is performed on our trained model for query and gallery.
```
python  feature_extraction.py [--ages_name args]
```

#### 4. Feature Enhancement
Feature enhancement is performed separately for the trained model.We use the pyretri repository（https://github.com/PyRetri/PyRetri/tree/master/pyretri）
```
python post_process.py [--ages_name args]
```

#### 5. Model Fusion
Fusion of the above trained models.
```
python model_fuse.py
```

#### 6. Similarity matrix post-processing
In this step the queries and galleries are grouped according to the similarity matrix, and the same set of queries is a class.
```
python post_based_sim.py [--ages_name args]
```
#### 7. Generate hashcode
Use the MD5 encryption method of hashlib to generate a 12-bit hexadecimal code for the image, and then convert it to a 48-bit hashcode.
```
python hash_code.py
```

