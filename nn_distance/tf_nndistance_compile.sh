/usr/local/cuda/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I /home/tj258/anaconda3/envs/py36/lib/python3.6/site-packages/tensorflow/include/ -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 && g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /home/tj258/anaconda3/envs/py36/lib/python3.6/site-packages/tensorflow/include/ -L /usr/local/cuda/lib64 -O2