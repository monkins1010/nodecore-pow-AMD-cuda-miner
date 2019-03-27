/usr/local/cuda/bin/nvcc  -gencode=arch=compute_52,code=\"sm_52,compute_52\" -I/usr/local/cuda/include -I. -O3 -Xcompiler -Wall  -D_FORCE_INLINES  --ptxas-options="-v" --maxrregcount=64 -o nodecore_pow_cuda kernel.cu main.cpp Miner.cpp -Xcompiler -static-libgcc -Xcompiler -static-libstdc++ -std=c++11

