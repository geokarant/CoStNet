# Detect-With-Style-A-Contrastive-Learning-Framework-For-Detecting-Computer-Generated-Images

This is the official repository of the paper "Detect With Style: A Contrastive Learning Framework For Detecting Computer Generated Images" [[paper]](https://www.mdpi.com/2078-2489/15/3/158).

```
@article{karantaidis2024detect,
  title={Detect with Style: A Contrastive Learning Framework for Detecting Computer-Generated Images},
  author={Karantaidis, Georgios and Kotropoulos, Constantine},
  journal={Information},
  volume={15},
  number={3},
  pages={158},
  year={2024},
  publisher={MDPI}
}
```


## Abstract
The detection of computer-generated (CG) multimedia content has become of utmost importance due to the advances in digital image processing and computer graphics. Realistic CG images could be used for fraudulent purposes due to the deceiving recognition capabilities of human eyes. So, there is a need to deploy algorithmic tools for distinguishing CG images from natural ones within multimedia forensics. Here, an end-to-end framework is proposed to tackle the problem of distinguishing CG images from natural ones by utilizing supervised contrastive learning and arbitrary style transfer by means of a two-stage deep neural network architecture. This architecture enables discrimination by leveraging per-class embeddings and generating multiple training samples to increase model capacity without the need for a vast amount of initial data. Stochastic weight averaging (SWA) is also employed to improve the generalization and stability of the proposed framework. Extensive experiments are conducted to investigate the impact of various noise conditions on the classification accuracy and the proposed framework’s generalization ability. The conducted experiments demonstrate superior performance over the existing state-of-the-art methodologies on the public DSTok, Rahmouni, and LSCGB benchmark datasets. Hypothesis testing asserts that the improvements in detection accuracy are statistically significant.



## Requirements

Create a virtual environmnet using the following:

```python
conda env create -f newenv.yml
```


# Learning module

## Training scripts

Stage 1:
 ```
python train.py --config_name configs/train_supcon_resnet18_DSTOK_stage1.yml
  ```
  ```
python swa.py --config_name configs/swa_supcon_resnet18_DSTOK_stage1.yml
  ```
Stage 2:
```
python train.py --config_name configs/train_supcon_resnet18_DSTOK_stage2.yml
  ```
  ```
python swa.py --config_name configs/swa_supcon_resnet18_DSTOK_stage2.yml
```

# Datasets
You should create and place your dataset under CoStNet/Learning module/data/photos/ folder. Your dataset under the photos/ folder should have the following structure:

    photos/
        test/
            CGG/
            Real/
        train/
            CGG/
            Real/

  In case you want to run your own dataset using a different structure, you can modify the tools/datasets.py file adding your own Class for you dataset and add your datast to 'DATASETS' dict inside tools/datasets.py. Pay attention to follow the same augmentation logic:
```
        if self.second_stage:
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)
```

  


# PAMA Style transfer module

The style transfer module can operate independently and doesn't require mandatory execution. You can run solely the learning module.


## Checkpoints

Please download the pre-trained checkpoints at [google drive](https://drive.google.com/file/d/1rPB_qnelVVSad6CtadmhRFi0PMI_RKdy/view?usp=sharing) and put them in ./checkpoints. 

Here we also provide some other pre-trained results with different loss weights:

| Type             | Loss            | Download             |
| ---------------- | --------------- | -------------------- |
| high consistency | w/o color loss  | [PAMA_without_color.zip](https://drive.google.com/file/d/1IrggOiutiZceJCrEb24cLnBjeA5I3N1D/view?usp=sharing) |
| high color       | 1.5x color loss weight | [PAMA_1.5x_color.zip](https://drive.google.com/file/d/1HXet2u_zk2QCVM_z5Llg2bcfvvndabtt/view?usp=sharing)       |
| high content     | 1.5x content loss weight | [PAMA_1.5x_content.zip](https://drive.google.com/file/d/13m7Lb9xwfG_DVOesuG9PyxDHG4SwqlNt/view?usp=sharing)     |


## Training

The training set consists of two parts, the content images from COCO2014 and style images from Wikiart.

```python
python main.py train --lr 1e-4 --content_folder ./COCO2014 --style_folder ./Wikiart
```

## Testing

To test the code, you need to specify the path of the content image and the style image. 

```python
python main.py eval --content ./content/1.jpg --style ./style/1.jpg
```

If you want to do a batch operation for all pictures under the folder at one time, please execute the following code.

```python
python main.py eval --run_folder True --content ./content/ --style ./style/
```


## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [PAMA](https://github.com/luoxuan-cs/PAMA)
- [SupCon](https://github.com/ivanpanshin/SupCon-Framework)

## If you find you find CoStNet useful, please also consider citing:

```
@inproceedings{luo2022progressive,
  title={Progressive attentional manifold alignment for arbitrary style transfer},
  author={Luo, Xuan and Han, Zhen and Yang, Linkang},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={3206--3222},
  year={2022}
}
```
# If you have any questions feel me to contant me directly at: gkarantai@csd.auth.gr
