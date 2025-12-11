![](./imgs/20251211125055.png)

<p align="center"> 
<!-- <a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320324002188" ><img src="https://img.shields.io/badge/HOME-PR-blue.svg"></a> -->
<a href="https://indtlab.github.io/projects/BADANet" ><img src="https://img.shields.io/badge/HOME-Paper-important.svg"></a>
<a href="https://INDTLab.github.io/projects/Packages/BADANet/Data/BADANet.pdf" ><img src="https://img.shields.io/badge/PDF-Paper-blueviolet.svg"></a>
<!-- <a href="https://indtlab.github.io/projects/WRD-Net" ><img src="https://img.shields.io/badge/-WeightsFiles-blue.svg"></a> -->
</p>

# Architecture

![](./imgs/20251211125141.png)

# Usage
### Installation
1. Create a virtual environment named `BADA`:   
```
conda create -n BADA python=3.8 -y
```     
2. Activate the new environment:  
```
conda activate BADA
```    
3. Install dependencies:  
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
```

### Data Sets
We provide data sets used in the paper: [CAMO](https://sites.google.com/view/ltnghia/research/camo), [CHAMELEON](https://www.polsl.pl/rau6/chameleon-database-animal-camouflage-analysis/), [COD0K](https://drive.google.com/file/d/1pVq1rWXCwkMbEZpTt4-yUQ3NsnQd_DNY/view?usp=sharing), [NC4K](https://drive.google.com/file/d/1kzpX_U3gbgO9MuwZIWTuRVpiB7V6yrAQ/view?usp=sharing)


### Pre-Trained Weights
You can download the pre-trained weights of BADANet from [GoogleLink](https://drive.google.com/file/d/1TSzoyMiNQPf13q9FbBCXSudZRAjdcB1N/view?usp=sharing)

### Train
You can train the model using the following command:  
```
python ./main.py \
--model-name CAMO \
--config "./configs/zoomnet/cod_zoomnet.py" \
--datasets-info "./configs/_base_/dataset/dataset_configs.json"
```
 ### Test
 You can evaluate the model on a specified dataset using the following command
 ```
python ./test.py \
 --model-name CAMO \
--config "./configs/zoomnet/cod_zoomnet.py" \
--datasets-info "./configs/_base_/dataset/dataset_configs.json" \
--load-from "./output/CAMO/best_ckpt/BADANet.pth" \
--save-path "./output/CAMO/test_result/CAMO
 ```

