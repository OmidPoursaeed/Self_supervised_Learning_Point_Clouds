nvcc=/usr/local/cuda/bin/nvcc
cudalib=/usr/local/cuda/lib64
nsync=/home/hq45/anaconda3/envs/condavenv/lib/python3.7/site-packages/tensorflow/include/external/nsync/public
TF_INC=/home/hq45/anaconda3/envs/condavenv/lib/python3.7/site-packages/tensorflow/include
TF_LIB=/home/hq45/anaconda3/envs/condavenv/lib/python3.7/site-packages/tensorflow/

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o

tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(TF_INC) -I $(nsync) -lcudart -L $(cudalib) -L $(TF_LIB) -l:libtensorflow_framework.so.1 -O2 -D_GLIBCXX_USE_CXX11_ABI=1

tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(TF_INC) -I $(nsync) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=1 --expt-relaxed-constexpr

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(TF_INC) -I $(nsync) -lcudart -L $(cudalib) -L $(TF_LIB) -l:libtensorflow_framework.so.1 -O2 -D_GLIBCXX_USE_CXX11_ABI=1

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(TF_INC) -I $(nsync) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=1 --expt-relaxed-constexpr

clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o