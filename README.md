# Structure Invariant Attack

This repository contains code to reproduce results from the paper:

[Structure Invariant Transformation for better Adversarial Transferability]() (ICCV 2023)

Xiaosen Wang, Zeliang Zhang, Jianping Zhang

![The transformed images generated by various transformations.](./figs/transformations.png)


## Requirements

+ Python >= 3.6.5
+ Numpy >= 1.15.4
+ opencv >= 3.4.2
+ scipy > 1.1.0
+ pandas >= 1.0.1
+ imageio >= 2.6.1
+ pytorch >= 1.14.0
+ torchvision >= 0.13

## Qucik Start

### Prepare the data and models

You should download the [data](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) offered by [Admix](https://github.com/JHL-HUST/Admix) and place it in Input/data with label file `val_rs.csv` in `Input`.

### Runing attack

Taking SIA attack for example, you can run this attack as following:

```
CUDA_VISIBLE_DEVICES=gpuid python main.py --model model_name  
```

### Evaluating the attack

The generated adversarial examples would be stored in directory `./outputs`. Then run the file `main.py` with `eval` to evaluate the success rate of each model used in the paper:

```
CUDA_VISIBLE_DEVICES=gpuid python main.py --eval
```

### Citation

If you find the idea or code useful for your research, please consider citing our [paper]():
```
@inproceedings{wang2023structure,
     title={{Structure Invariant Transformation for better Adversarial Transferability}},
     author={Xiaosen Wang and Zeliang Zhang and Jianping Zhang},
     booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
     year={2023}
}
```