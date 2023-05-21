python feature_extraction.py --model beit_large_patch16_512.in22k_ft_in22k_in1k\
                             --checkpoint /home/data1/changhao/iBioHash/Record/cls_baseline/beit/beit_size512/20230402-003503-beit_large_patch16_512_in22k_ft_in22k_in1k-512/model_best.pth.tar\
                             -b 64\
                             --num-gpu 1\
                             --amp-impl apex\
                             --feature_dim 1024\
                             --feature_outdir beit_512