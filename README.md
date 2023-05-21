# solution_iBioHash_2023_FGVC10_1st
Top 1 (1/23)  solution for [iBioHash 2023](https://www.kaggle.com/competitions/ibiohash-2023-fgvc10/overview) competition, which is as part of the  [FGVC10](https://sites.google.com/view/fgvc10/home) workshop at [CVPR 2023](http://cvpr2023.thecvf.com/)

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
We used both eva_large_patch14_336.in22k_ft_in1k and beit_large_patch16_512.in22k_ft_in22k_in1k models.
```
sh 
```
#### 3. Feature Extraction
Here the feature extraction is performed on our trained model for query and gallery and the similarity matrix between the features is calculated.
```
sh 
```
#### 4. Model Fusion
Fusion of the above trained models.
```
sh 
```

#### 5. Similarity matrix post-processing
In this step the queries and galleries are grouped according to the similarity matrix, and the same set of queries is a class.
```
sh 
```
#### 6. Generate hashcode
Use the MD5 encryption method of hashlib to generate a 12-bit hexadecimal code for the image, and then convert it to a 48-bit hashcode.
```
python
```

