# Semi Hand-Object
### Semi-Supervised 3D Hand-Object Poses Estimation with Interactions in Time (CVPR 2021). [![report](https://img.shields.io/badge/arxiv-report-red)]() [![Open In Google-Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

[Project Page with Videos]()
![Teaser](assets/figs/teaser.png)


## Installation
- Clone this repository: 
    ```Shell
    git clone https://github.com/stevenlsw/Semi-Hand-Object.git
    ```
- Install the dependencies by the following command:
    ```Shell
    pip install -r requirements.txt
    ```
  
## Quick Demo 
Quick demo on [google-colab]().


## Training and Evaluation on HO3D Dataset
### Preparation

- Download the MANO model files (`mano_v1_2.zip`) from [MANO website](http://mano.is.tue.mpg.de/). 
Unzip and put `mano/models/MANO_RIGHT.pkl` into `assets/mano_models`. 

- Download the [YCB-Objects](https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu) 
used in [HO3D dataset](https://www.tugraz.at/index.php?id=40231). Put unzipped folder `object_models` under `assets`.

- The structure should look like this:

```
Semi-Hand-Object/
  assets/
    mano_models/
      MANO_RIGHT.pkl
    object_models/
      006_mustard_bottle/
        points.xyz
        textured_simple.obj
      ......
```
- Download and unzip [HO3D dataset](https://cloud.tugraz.at/index.php/s/9HQF57FHEQxkdcz/download?path=%2F&files=HO3D_v2.zip) 
to path you like, the unzipped path is referred as `$HO3D_root`.  


### Evaluation 
The hand & object pose estimation performance on HO3D dataset. 
We evaluate hand pose results on the official [CodaLab challenge](https://competitions.codalab.org/competitions/22485?). 
The hand metric below is mean joint/mesh error after procrustes alignment, 
the object metric is average object vertices error within 10% of object diameter (ADD-0.1D). 

The model `earlier` and `latest` are different in CR module, the former is used in the 
conference paper, while the latter is used in arxiv version. 
In our latest model, we use **transformer** architecture to perform hand-object contextual reasoning.

Please download the trained model and save to path you like, the model path is refered as `$resume`.

| model   | link         | joint↓ | mesh↓ | cleanser↑ | bottle↑ | can↑ | ave↑ |
|---------|--------------|--------|-------|-----------|---------|------|------|
| earlier | [download]() |  0.98  |  0.94 |    91.6   |   73.1  | 59.2 | 74.6 |
|  latest | [download]() |  0.99  |  0.95 |    92.2   |   80.4  | 55.7 | 76.1 |


- #### Testing with latest model
 ```
    python traineval.py --evaluate --HO3D_root={path to the dataset} --resume={path to the model} --test_batch=24 --host_folder=exp_results
 ```

- #### Testing with earlier model
```
   python traineval.py --evaluate --HO3D_root={path to the dataset} --resume={path to the model} --network=honet_attention --test_batch=24 --host_folder=exp_results
```

The testing results will be saved in the `$host_folder`, which contains the following files: 
* `option.txt` (saved options) 
* `object_result.txt` (object pose evaluation performance) 
* `pred.json` (```zip -j pred.zip pred.json``` and submit to the [offical challenge](https://competitions.codalab.org/competitions/22485?) for hand evaluation)


### Training
Please download the [preprocessed files]() to train HO3D dataset. 
The downloaded files contains training list and labels generated from the original dataset to accelerate training. 
Please put the unzipped folder `ho3d-process` to current directory.  

```
    python traineval.py --HO3D_root={path to the dataset} --train_batch=24 --host_folder=exp_results
```
The models will be automatically saved in `$host_folder`


## Citation
```
@inproceedings{liu2021semi,
  title={Semi-Supervised 3D Hand-Object Poses Estimation with Interactions in Time},
  author={Liu, Shaowei and Jiang, Hanwen and Xu, Jiarui and Liu, Sifei and Wang, Xiaolong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```

## TODO
- [x] Google colab demo
- [ ] Minimal implementation of pseudo-label generation and filtering


## Acknowledgments
We thank: 
* [obman_train](https://github.com/hassony2/obman_train.git) provided by Yana Hasson
* [segmentation-driven-pose](https://github.com/cvlab-epfl/segmentation-driven-pose.git) provided by 
Yinlin Hu