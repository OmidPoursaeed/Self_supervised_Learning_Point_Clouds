# Self-supervised learning of point clouds via orientation estimation

## Installation

We use tensorflow 1.14, python 3.6 and h5py.

## Usage


To train the rotation prediction model with 18 random rotation angles, run:

```
python train_rotation_prediction.py --num_angles 18 --dataset shapenet --log_dir log_rotation_shapenet --no_transformation_loss --no_input_transform --no_feature_transform;
```

Please see `train_rotation_prediction.py` for more arguments.

Then, to train a linear SVM for object classification on top of the pretrained weights, run

```
python SVM.py --model pointnet_cls_rot_svm_scoped --svm_c 0.001 --dataset modelnet10 --model_path PATH/TO/CHECKPOINT
```

## Datasets

The ModelNet data will be automatically downloaded to `data/` directory

The ShapeNet data can be downloaded from [here](https://www.shapenet.org/). Then set the `SHAPENET_DIR` in `provider.py` to the ShapeNet folder.
