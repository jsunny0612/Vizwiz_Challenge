# [CVPR Vizwiz Challenge 2024]
## Environments
+ Ubuntu 18.04.6 LTS
+ CUDA Version 11.3
+ GPU: NVIDIA RTX A6000 with 48GB memory

## Prerequisites

To use the repository, we provide a conda environment.

```bash
conda update conda
conda env create -f environment.yaml
conda activate vizwiz_TTA
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Structure of Project

This project contains several directories.

Please note that the dataset is 8900 images.

+ ./best_cfgs: the best config files for each dataset and algorithm are saved here.
+ ./datasets
  
  	    |-- datasets 
  	
  	        |-- challenge
  	
  	                |-- original
  	
  	                        |-- 5
  	
  	                              |-- vizwiz

  
## Get Started

Specify the root folder for all datasets `_C.DATA_DIR = "./datasets"` in the file `conf.py`.

Please change the number of contrasts in ./best_sfgs/parallel_psedo_contrast.yaml.

## How to reproduce

Train and test model

    CUDA_VISIBLE_DEVICES=0,1,2,3 python challenge_test_time.py --cfg ./best_cfgs/parallel_psedo_contrast.yaml --output_dir ./test-time-evaluation/"[YOUR EXPERIMENRT NAME]"

The testing results and training logs will be saved in the `./output/test-time-evaluation/"[YOUR EXPERIMENRT NAME]"`

## Acknowledgements

Our codes borrowed from [yuyongcan](https://github.com/yuyongcan/Benchmark-TTA). Thanks for their work.
