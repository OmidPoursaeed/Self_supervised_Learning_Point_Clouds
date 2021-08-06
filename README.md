# Self supervised Learning of Point Clouds via Orientation Estimation

Code for the paper "[Self-supervised Learning of Point Clouds via Orientation Estimation](https://arxiv.org/pdf/2008.00305.pdf)", 3DV 2020.

## Installation

We use tensorflow 1.14, python 3.6 and h5py.

## Usage

To train the rotation prediction model with 18 random rotation angles run:

```
python train_rotation_prediction.py --num_angles 18 --dataset shapenet --log_dir log_rotation_shapenet --no_transformation_loss --no_input_transform --no_feature_transform;
```

Please see `train_rotation_prediction.py` for more arguments.

[Here](https://drive.google.com/drive/folders/1NewPE6b7MdK1PyJ-P9JWo2RLIdQu_-NU?usp=sharing) is a pretrained model of running the script above.

To train a linear SVM for object classification on top of the pretrained weights run:

```
python SVM.py --model pointnet_cls_rot_svm_scoped --svm_c 0.001 --dataset modelnet10 --model_path PATH/TO/CHECKPOINT
```

## Datasets

The ModelNet data will be automatically downloaded to the `data/` directory.

The ShapeNet data can be downloaded from [here](https://www.shapenet.org/). Set `SHAPENET_DIR` in `provider.py` to the ShapeNet folder.

## Citation

If you use the code in this repository in your paper, please consider citing:
```
@article{poursaeed2020self,
  title={Self-supervised Learning of Point Clouds via Orientation Estimation},
  author={Poursaeed, Omid and Jiang, Tianxing and Qiao, Quintessa and Xu, Nayun and Kim, Vladimir G.},
  journal={arXiv preprint arXiv:2008.00305},
  year={2020}
}
```

