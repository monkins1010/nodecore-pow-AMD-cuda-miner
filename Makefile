# Change this path if the SDK was installed in a non-standard location
OPENCL_HEADERS = "/opt/AMDAPPSDK-3.0/include"
# By default libOpenCL.so is searched in default system locations, this path
# lets you adds one more directory to the search path.
LIBOPENCL = "/opt/amdgpu-pro/lib/x86_64-linux-gnu"

CC = g++
CPPFLAGS = -I${OPENCL_HEADERS}
CFLAGS = -O2 -Wall
LDFLAGS =  -pthread -lpthread -rdynamic -L${LIBOPENCL} -fpermissive
LDLIBS = -lOpenCL -lrt
OBJ = kernel.cc
INCLUDES = _kernel.h

all : veri_amd

veri_amd : ${OBJ}
	${CC} -std=gnu++11 -o veri_amd ${OBJ} ${LDFLAGS} ${LDLIBS}

${OBJ} : ${INCLUDES}

_kernel.h : input.cl 
	echo 'const char *ocl_code = R"_mrb_(' >$@
	cpp $< >>$@
	echo ')_mrb_";' >>$@

clean :
	rm -f veri_amd _kernel.h *.o _temp_*

re : clean all
