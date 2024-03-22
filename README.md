# Detect-With-Style-A-Contrastive-Learning-Framework-For-Detecting-Computer-Generated-Images

This is the official repository of the paper "Detect With Style: A Contrastive Learning Framework For Detecting Computer Generated Images"

## Code will be soon available!


## Requirements

Create the appropriate environmnet using the following:

```python
conda env create -f newenv.yml
```



## PAMA Style transfer module


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

## Learning module

