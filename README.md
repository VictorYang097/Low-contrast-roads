# Low-contrast-roads
This repository contains codes for **LCMorph** and access links to the **LC-Roads dataset**.

## LCMorph
In order to train LCMorph on the **DeepGlobe dataset** from the scratch, you can run:
```bash
python MyTrain_Models_DP.py --epoch=100 --batch_size=8 --model_name=LCMorph --no_improve=10 --resume=False
```
In order to continue training LCMorph on the **DeepGlobe dataset**, you can run:
```bash
python MyTrain_Models_DP.py --epoch=100 --batch_size=8 --model_name=LCMorph --no_improve=10 --resume=True
```
---
In order to train LCMorph on the **Massachusetts Roads dataset** from the scratch, you can run:
```bash
python MyTrain_Models_Mass.py --epoch=120 --batch_size=8 --model_name=LCMorph --no_improve=10 --resume=False
```
In order to continue training LCMorph on the **Massachusetts Roads dataset**, you can run:
```bash
python MyTrain_Models_Mass.py --epoch=120 --batch_size=8 --model_name=LCMorph --no_improve=10 --resume=True
```
---
In order to train LCMorph on our **LC-Roads dataset** from the scratch, you can run:
```bash
python MyTrain_Models_LC.py --epoch=80 --batch_size=8 --model_name=LCMorph --no_improve=8 --resume=False
```
In order to continue training LCMorph on our **LC-Roads dataset**, you can run:
```bash
python MyTrain_Models_LC.py --epoch=80 --batch_size=8 --model_name=LCMorph --no_improve=8 --resume=True
```
## LC-Roads dataset
**LC-Roads** is a dataset for low-contrast road extraction.

We constructed LC-Roads dataset based on the DeepGlobe dataset.  
For DeepGlobe, you can find the dataset at [DeepGlobe-CVPR2018](http://deepglobe.org/challenge.html).  
The related paper is at [here](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Demir_DeepGlobe_2018_A_CVPR_2018_paper.pdf).
Thanks a lot for their great work!

For our LC-Roads dataset, you can find it at [Baidu Drive](https://pan.baidu.com/s/1gsIckLO9shqfAzznQizZKA?pwd=fcrd) (download code: fcrd), and [Google Drive](https://drive.google.com/file/d/1Hxccb6TCjvlVL0UphVJAtLzGs4TIQWRt/view?usp=sharing).
