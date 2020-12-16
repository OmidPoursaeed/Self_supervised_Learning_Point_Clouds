python train_rotation_regression.py --dataset modelnet --log_dir temp --no_transformation_loss --no_input_transform --no_feature_transform;

# 6d representation
python train_rotation_regression.py --model pointnet_reg_rotation6d --dataset modelnet --log_dir log_temp_6d --no_transformation_loss --no_input_transform --no_feature_transform;