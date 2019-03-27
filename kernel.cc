#define NOMINMAX
#include <cstdint>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <CL/cl.h>
#include <chrono>
#include <ctime>
#include <pthread.h>
//#include <fcntl.h>
//#include <unistd.h>
//#include <getopt.h>
//#include <errno.h>
//#include <time.h>
//#include <stdio.h>
//#include <unistd.h>
#include <stdlib.h>
#include "UCPClient.h"
#include <fcntl.h>
//#include <stddef.h>

#ifdef _WIN32
typedef SSIZE_T ssize_t;
#include <Windows.h>
#include <VersionHelpers.h>
#include <io.h>
#include <BaseTsd.h>
#endif

#ifdef __linux__
#include <sys/socket.h> 
#include <netdb.h>
#include "_kernel.h"
#endif

#include <ctime>
#include "Log.h"
#include <sstream>
#include "Constants.h"

#ifndef O_BINARY
#define O_BINARY 0
#endif

#define MAX_GPUS 16
#define DEFAULT_BLOCKSIZE 0x2000
#define DEFAULT_THREADS_PER_BLOCK 256
#define WORK_PER_THREAD 128

pthread_mutex_t stratum_sock_lock;
pthread_mutex_t stratum_log_lock;
cl_device_id *devices;
int blocksize = DEFAULT_BLOCKSIZE;
int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
int opt_platform_id = -1;
bool verboseOutput = false;
int amd_flag = 0;

//#define open _open
const char *source = NULL;
size_t source_len;
char *binary = NULL;
size_t binary_len;
uint32_t lastNonceStart = 0;
char outputBuffer[8192];
char selectedDeviceName[100];

//amd stuff

cl_platform_id platform_M = NULL;
cl_program program[MAX_GPUS];
cl_command_queue queue[MAX_GPUS];
cl_context context[MAX_GPUS];
cl_device_id	dev_id[MAX_GPUS] = { 0 };
cl_kernel k_vblake[MAX_GPUS];
UCPClient* pUCP;

struct mining_attr {
	int dev_id;

};

int opt_n_threads = 0;
short device_map[MAX_GPUS] = { 0 };
int gpu_threads = 1;
int active_gpus;
char device_name[MAX_GPUS][256];
long  device_sm[MAX_GPUS] = { 0 };
short device_mpcount[MAX_GPUS] = { 0 };
int init[MAX_GPUS] = { 0 };
uint32_t lastnonce[4] = { 6,7,8,9 };


/*
* Kernel function to search a range of nonces for a solution falling under the macro-configured difficulty (CPU=2^24, GPU=2^32).
*/


void promptExit(int exitCode)
{
	cout << "Exiting in 10 seconds..." << endl;
	std::this_thread::sleep_for(std::chrono::milliseconds(10000));
	exit(exitCode);
}

/**
* Takes the provided timestamp and places it in the header
*/
void embedTimestampInHeader(uint8_t *header, uint32_t timestamp)
{
	header[55] = (timestamp & 0x000000FF);
	header[54] = (timestamp & 0x0000FF00) >> 8;
	header[53] = (timestamp & 0x00FF0000) >> 16;
	header[52] = (timestamp & 0xFF000000) >> 24;
}

/**
* Returns a 64-byte header to attempt to mine with.
*/
void getWork(UCPClient& ucpClient, uint32_t timestamp, uint64_t *header)
{
	//uint64_t *header = new uint64_t[8];
	//printf("time stamp %08x \n", timestamp);
	ucpClient.copyHeaderToHash((byte *)header);
	embedTimestampInHeader((uint8_t*)header, timestamp);
	//return header;
}

int deviceToUse = 0;

#if NVML
nvmlDevice_t device;
void readyNVML(int deviceIndex) {
	nvmlInit();
	nvmlDeviceGetHandleByIndex(deviceIndex, &device);
}
int getTemperature() {
	unsigned int temperature;
	nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
	return temperature;
}

int getCoreClock() {
	unsigned int clock;
	nvmlDeviceGetClock(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &clock);
	return clock;
}

int getMemoryClock() {
	unsigned int memClock;
	nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &memClock);
	return memClock;
}
#else
void readyNVML(int deviceIndex) {
	// Do Nothing
}

int getTemperature() {
	return -1;
}

int getCoreClock() {
	return -1;
}

int getMemoryClock() {
	return -1;
}
#endif

#define SHARE_SUBMISSION_NO_RESPONSE_WARN_THRESHOLD 50

void vprintf(char* toprint) {
	if (verboseOutput) {
		printf(toprint);
	}
}

void printHelpAndExit() {
	printf("VeriBlock vBlake OpenCL Miner v1.0\n");
	printf("Required Arguments:\n");
	printf("-o <poolAddress>           The pool address to mine to in the format host:port\n");
	printf("-u <username>              The username (often an address) used at the pool\n");
	printf("Optional Arguments:\n");
	printf("-p <password>              The miner/worker password to use on the pool\n");
	printf("-d <deviceNum>             The ordinal of the device to use (default 0)\n");
	printf("-tpb <threadPerBlock>      The threads per block to use with the Blake kernel (default %d)\n", DEFAULT_THREADS_PER_BLOCK);
	printf("-bs <blockSize>            The blocksize to use with the vBlake kernel (default %d)\n", DEFAULT_BLOCKSIZE);
	printf("-l <enableLogging>         Whether to log to a file (default true)\n");
	printf("-v <enableVerboseOutput>   Whether to enable verbose output for debugging (default false)\n");
	printf("\n");
	printf("Example command line:\n");
	printf("VeriBlock-NodeCore-PoW-CUDA -u VHT36jJyoVFN7ap5Gu77Crua2BMv5j -o testnet-pool-gpu.veriblock.org:8501 -l false\n");
	promptExit(0);
}

#ifdef _WIN32
static WSADATA g_wsa_data;
#endif

char net_init(void)
{
#ifdef _WIN32
	return (WSAStartup(MAKEWORD(2, 2), &g_wsa_data) == NO_ERROR);
#elif __linux__
	return 1;
#endif
}

void net_deinit(void)
{
#ifdef _WIN32
	WSACleanup();
#endif
}

string net_dns_resolve(const char* hostname)
{
	struct addrinfo hints, *results, *item;
	int status;
	char ipstr[INET6_ADDRSTRLEN];

	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC;  /* AF_INET6 to force version */
	hints.ai_socktype = SOCK_STREAM;

	if ((status = getaddrinfo(hostname, NULL, &hints, &results)) != 0)
	{
		fprintf(stderr, "failed to resolve hostname \"%s\": %s", hostname, gai_strerror(status));
		return "invalid hostname";
	}

	printf("IP addresses for %s:\n\n", hostname);

	string ret;

	for (item = results; item != NULL; item = item->ai_next)
	{
		void* addr;
		char* ipver;

		/* get pointer to the address itself */
		/* different fields in IPv4 and IPv6 */
		if (item->ai_family == AF_INET)  /* address is IPv4 */
		{
			struct sockaddr_in* ipv4 = (struct sockaddr_in*)item->ai_addr;
			addr = &(ipv4->sin_addr);
			ipver = "IPv4";
		}
		else  /* address is IPv6 */
		{
			struct sockaddr_in6* ipv6 = (struct sockaddr_in6*)item->ai_addr;
			addr = &(ipv6->sin6_addr);
			ipver = "IPv6";
		}

		/* convert IP to a string and print it */
		inet_ntop(item->ai_family, addr, ipstr, sizeof ipstr);
		printf("  %s: %s\n", ipver, ipstr);
		ret = ipstr;
	}

	freeaddrinfo(results);
	return ret;
}
cl_mem check_clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size,
	void *host_ptr)
{
	cl_int	status;
	cl_mem	ret;
	ret = clCreateBuffer(ctx, flags, size, host_ptr, &status);
	if (status != CL_SUCCESS || !ret) {
		sprintf(outputBuffer, "clCreateBuffer (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}
	//fatal("clCreateBuffer (%d)\n", status);
	return ret;
}

void dump(const char *fname, void *data, size_t len)
{
	int			fd;
	ssize_t		ret;
	if (-1 == (fd = open(fname, O_BINARY | O_WRONLY | O_CREAT | O_TRUNC, 0666))) {
		sprintf(outputBuffer, "%s: %s", fname, strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);

	}

	ret = write(fd, data, len);
	if (ret == -1) {
		sprintf(outputBuffer, "write: %s: %s", fname, strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);

	}

	if ((size_t)ret != len) {
		sprintf(outputBuffer, "%s: partial write", fname, strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);

	}

	if (-1 == close(fd)) {
		sprintf(outputBuffer, "close: %s: %s", fname, strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);

	}

}

void get_program_bins(cl_program program)
{
	cl_int		status;
	size_t		sizes;
	unsigned char	*p;
	size_t		ret = 0;
	status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
		sizeof(sizes),	// size_t param_value_size
		&sizes,		// void *param_value
		&ret);		// size_t *param_value_size_ret

	p = (unsigned char *)malloc(sizes);
	status = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
		sizeof(p),	// size_t param_value_size
		&p,		// void *param_value
		&ret);	// size_t *param_value_size_ret

	dump("dump.co", p, sizes);

}
void print_platform_info(cl_platform_id plat)
{
	char	name[1024];
	size_t	len = 0;
	int		status;
	status = clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(name), &name,
		&len);
	if (status != CL_SUCCESS)
	{
		sprintf(outputBuffer, "clGetDeviceInfo (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);

	}
	sprintf(outputBuffer, "Devices on platform \"%s\":", name);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	//printf("Devices on platform \"%s\":\n", name);
	//fflush(stdout);
}
void print_device_info(unsigned i, cl_device_id d)
{
	char	name[1024];
	size_t	len = 0;
	int		status;
	status = clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), &name, &len);
	if (status != CL_SUCCESS) {
		sprintf(outputBuffer, "clGetDeviceInfo (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
		//fatal("malloc: %s\n", strerror(errno));
	}
	strcpy(selectedDeviceName,name);
	sprintf(outputBuffer, "  ID %d: %s", i, name);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);


	//fatal("clGetDeviceInfo (%d)\n", status);
	//printf("  ID %d: %s\n", i, name);

}
void check_clEnqueueReadBuffer(cl_command_queue queue, cl_mem buffer, cl_bool
	blocking_read, size_t offset, size_t size, void *ptr, cl_uint
	num_events_in_wait_list, const cl_event *event_wait_list, cl_event
	*event)
{
	cl_int	status;
	status = clEnqueueReadBuffer(queue, buffer, blocking_read, offset,
		size, ptr, num_events_in_wait_list, event_wait_list, event);
	if (status != CL_SUCCESS) {
		sprintf(outputBuffer, "clEnqueueReadBuffer (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

}
void check_clSetKernelArg(cl_kernel k, cl_uint a_pos, cl_mem *a)
{
	cl_int	status;
	status = clSetKernelArg(k, a_pos, sizeof(*a), a);
	if (status != CL_SUCCESS) {
		sprintf(outputBuffer, "clSetKernelArg (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

}
void check_clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel k, cl_uint
	work_dim, const size_t *global_work_offset, const size_t
	*global_work_size, const size_t *local_work_size, cl_uint
	num_events_in_wait_list, const cl_event *event_wait_list, cl_event
	*event)
{
	cl_uint	status;
	status = clEnqueueNDRangeKernel(queue, k, work_dim, global_work_offset,
		global_work_size, local_work_size, num_events_in_wait_list,
		event_wait_list, event);
	if (status != CL_SUCCESS) {
		sprintf(outputBuffer, "clEnqueueNDRangeKernel (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

}


void get_program_build_log(cl_program program, cl_device_id device)
{
	cl_int		status2;
	char	        val[100 * 1024];
	size_t		ret = 0;

	status2 = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(val), &val, &ret);
	// if (status2 != CL_SUCCESS) {
	sprintf(outputBuffer, "%s", val);

	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	//promptExit(-1);
	//}

}

int is_platform_amd(cl_platform_id plat)
{
	char	name[1024];
	size_t	len = 0;
	int		status;
	status = clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(name), &name,
		&len);
	if (status != CL_SUCCESS)
	{
		sprintf(outputBuffer, "clGetPlatformInfo (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}
	return strncmp(name, "AMD Accelerated Parallel Processing", len) == 0;
}

#ifdef _WIN32
void load_file(const char *fname, char **dat, size_t *dat_len, int ignore_error)
{
	struct stat	st;
	int		fd;
	ssize_t	ret;
	if (-1 == (fd = open(fname, O_RDONLY | O_BINARY))) {
		if (ignore_error)
			return;
		//fatal("%s: %s\n", fname, strerror(errno));
		sprintf(outputBuffer, "%s: %s", fname, strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}
	if (fstat(fd, &st)) {
		sprintf(outputBuffer, "fstat: %s: %s", fname, strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	*dat_len = st.st_size;
	if (!(*dat = (char *)malloc(*dat_len + 1))) {
		sprintf(outputBuffer, "malloc: %s", strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	ret = read(fd, *dat, *dat_len);
	if (ret < 0) {
		sprintf(outputBuffer, "read: %s: %s", fname, strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}
	if ((size_t)ret != *dat_len) {
		sprintf(outputBuffer, "%s: partial read", fname);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	if (close(fd)) {
		sprintf(outputBuffer, "close: %s: %s", fname, strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}
	(*dat)[*dat_len] = 0;
}
#endif

bool get_opencl_platform(int preferred_platform_id, cl_platform_id *platform) {
	cl_int status;
	cl_uint numPlatforms;
	cl_platform_id *platforms = NULL;
	char pbuff[256];
	unsigned int i;
	bool ret = false;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	/* If this fails, assume no GPUs. */
	if (status != CL_SUCCESS) {
		printf("Error %d: clGetPlatformsIDs failed (no OpenCL SDK installed?)", status);
		goto out;
	}

	if (numPlatforms == 0) {
		printf("clGetPlatformsIDs returned no platforms (no OpenCL SDK installed?)");
		goto out;
	}

	if (preferred_platform_id >= (int)numPlatforms) {
		printf("Specified platform that does not exist");
		goto out;
	}

	platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		printf("Error %d: Getting Platform Ids. (clGetPlatformsIDs)", status);
		goto out;
	}

	for (i = 0; i < numPlatforms; i++) {
		if (preferred_platform_id >= 0 && (int)i != preferred_platform_id)
			continue;

		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL);
		if (status == CL_SUCCESS && !strstr(pbuff, "Intel")) {

			printf("%d: %s\n", i, pbuff);
			*platform = platforms[i];
			ret = true;
			break;
		}
	}

out:
	if (platforms) free(platforms);
	return ret;
}

int clDevicesNum(void) {
	cl_int status;
	char pbuff[256];
	cl_uint numDevices;

	int ret = -1;

	if (!get_opencl_platform(opt_platform_id, &platform_M)) {
		goto out;
	}

	status = clGetPlatformInfo(platform_M, CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL);
	if (status != CL_SUCCESS) {
		printf("Error %d: Getting Platform Info. (clGetPlatformInfo)", status);
		goto out;
	}
	bool IntelPlatform = strstr(pbuff, "Intel");
	if (IntelPlatform)
		goto out;  // don't try to run on Intel integrated gpu


	printf("CL Platform vendor: %s\n", pbuff);
	status = clGetPlatformInfo(platform_M, CL_PLATFORM_NAME, sizeof(pbuff), pbuff, NULL);
	if (status == CL_SUCCESS)
		printf("CL Platform name: %s\n", pbuff);
	status = clGetPlatformInfo(platform_M, CL_PLATFORM_VERSION, sizeof(pbuff), pbuff, NULL);
	if (status == CL_SUCCESS)
		printf("CL Platform version: %s\n", pbuff);
	status = clGetDeviceIDs(platform_M, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (status != CL_SUCCESS) {
		printf("Error %d: Getting Device IDs (num)\n", status);
		goto out;
	}
	printf("Platform devices: %d\n", numDevices);
	if (numDevices) {
		unsigned int j;
		devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));

		clGetDeviceIDs(platform_M, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		for (j = 0; j < numDevices; j++) {
			dev_id[j] = devices[j];
			clGetDeviceInfo(dev_id[j], CL_DEVICE_NAME, sizeof(pbuff), pbuff, NULL);
			strcpy(device_name[j], pbuff);
			printf("\t%i\t%s\n", j, pbuff);
		}

	}

	ret = numDevices;
out:
	return ret;
}



void* miner_thread(void* arg) {
	struct mining_attr *arg_Struct =
		(struct mining_attr*) arg;
	short thr_id = arg_Struct->dev_id;
	uint32_t end_nonce = 0x20000000u * (thr_id + 1);
	// Run initialization of device before beginning timer
	uint64_t header[8];
	pthread_mutex_lock(&stratum_sock_lock);
	getWork(*pUCP, (uint32_t)std::time(0), header);
	pthread_mutex_unlock(&stratum_sock_lock);
	pthread_mutex_lock(&stratum_log_lock);
	unsigned long long startTime = std::time(0);
	pthread_mutex_unlock(&stratum_log_lock);

	uint32_t nonceResult[1] = { 0 };
	uint64_t hashStart[1] = { 0 };
	uint32_t startNonce = 0x20000000u * thr_id;
	size_t		global_ws;
	unsigned long long hashes = 0;
	size_t  local_work_size = (unsigned int)threadsPerBlock;
	uint64_t  *phashstartout, *pheaderin;
	uint32_t  *pnoncestart, *pnonceout;
	cl_mem   pnoncestart_d, pnonceout_d, phashstartout_d, pheaderin_d;
	//cudaError_t cudaStatus;  replaced with status
	pnoncestart = (uint32_t*)malloc(sizeof(uint32_t) * 2);
	pnonceout = (uint32_t *)malloc(sizeof(uint32_t) * 2);
	phashstartout = (uint64_t *)malloc(sizeof(uint64_t) * 2);
	pheaderin = (uint64_t *)malloc(sizeof(uint64_t) * 9);

	pnoncestart_d = check_clCreateBuffer(context[thr_id], CL_MEM_READ_WRITE, sizeof(uint32_t) * 1, NULL);
	pnonceout_d = check_clCreateBuffer(context[thr_id], CL_MEM_READ_WRITE, sizeof(uint32_t) * 1, NULL);
	phashstartout_d = check_clCreateBuffer(context[thr_id], CL_MEM_READ_WRITE, sizeof(uint64_t) * 1, NULL);
	pheaderin_d = check_clCreateBuffer(context[thr_id], CL_MEM_READ_WRITE, sizeof(uint64_t) * 8, NULL);

	uint32_t count = 0;

	int numLines = 0;
	printf("got in thread innit\n");
	// Mining loop
	while (true) {
		vprintf("top of mining loop\n");
		count++;
		long timestamp = (long)std::time(0);
		//delete[] header;
		vprintf("Getting work...\n");
		pthread_mutex_lock(&stratum_sock_lock);
		getWork(*pUCP, timestamp, header);
		vprintf("Getting job id...\n");
		int jobId = pUCP->getJobId();
		pthread_mutex_unlock(&stratum_sock_lock);
		count++;
		vprintf("Running kernel...\n");
		for (int i = 0; i<8; i++)
			pheaderin[i] = header[i];
		pnonceout[0] = 0;
		clEnqueueWriteBuffer(queue[thr_id], pnonceout_d, CL_TRUE, 0, sizeof(uint32_t) * 1, pnonceout, 0, NULL, NULL);
		phashstartout[0] = 0;
		clEnqueueWriteBuffer(queue[thr_id], pheaderin_d, CL_TRUE, 0, sizeof(uint64_t) * 8, pheaderin, 0, NULL, NULL);


		uint32_t nonceStart = (uint64_t)lastNonceStart + (blocksize * 128 * threadsPerBlock);
		lastNonceStart = nonceStart;
		pnoncestart[0] = startNonce;

		clEnqueueWriteBuffer(queue[thr_id], pnoncestart_d, CL_TRUE, 0, sizeof(uint32_t) * 1, pnoncestart, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue[thr_id], phashstartout_d, CL_TRUE, 0, sizeof(uint64_t) * 1, phashstartout, 0, NULL, NULL);

		check_clSetKernelArg(k_vblake[thr_id], 0, &pnoncestart_d);
		check_clSetKernelArg(k_vblake[thr_id], 1, &pnonceout_d);
		check_clSetKernelArg(k_vblake[thr_id], 2, &phashstartout_d);
		check_clSetKernelArg(k_vblake[thr_id], 3, &pheaderin_d);
		vprintf("Wrote buffers...\n");
		global_ws = (unsigned int)(blocksize * threadsPerBlock * 128);

		//start Opencl kernel
		check_clEnqueueNDRangeKernel(queue[thr_id], k_vblake[thr_id], 1, NULL,
			&global_ws, &local_work_size, 0, NULL, NULL);
		clFinish(queue[thr_id]);

		check_clEnqueueReadBuffer(queue[thr_id], pnonceout_d,
			CL_TRUE,	// cl_bool	blocking_read
			0,		// size_t	offset
			sizeof(uint32_t) * 1,	// size_t	size
			pnonceout,	// void		*ptr
			0,		// cl_uint	num_events_in_wait_list
			NULL,	// cl_event	*event_wait_list
			NULL);	// cl_event	*event

		check_clEnqueueReadBuffer(queue[thr_id], phashstartout_d,
			CL_TRUE,	// cl_bool	blocking_read
			0,		// size_t	offset
			sizeof(uint64_t) * 1,	// size_t	size
			phashstartout,	// void		*ptr
			0,		// cl_uint	num_events_in_wait_list
			NULL,	// cl_event	*event_wait_list
			NULL);	// cl_event	*event

					//	cl_int openclStatus = grindNonces(nonceResult, hashStart, header, k_vblake, queue, context, dev_id, program);

		nonceResult[0] = pnonceout[0];

		hashStart[0] = phashstartout[0];
		pthread_mutex_lock(&stratum_log_lock);
		unsigned long long totalTime = std::time(0) - startTime;
		pthread_mutex_unlock(&stratum_log_lock);
		hashes += ((uint32_t)blocksize * (uint32_t)threadsPerBlock * WORK_PER_THREAD);
		if ((uint64_t)startNonce + (uint64_t)(blocksize * threadsPerBlock * WORK_PER_THREAD) < (uint64_t)end_nonce) {
			startNonce += ((uint32_t)blocksize * (uint32_t)threadsPerBlock * WORK_PER_THREAD);
		}
		else
			startNonce = 0x20000000u * (uint32_t)thr_id;

		double hashSpeed = (double)hashes;
		hashSpeed /= (totalTime * 1024 * 1024);

		if (count % 20 == 0) {
			pthread_mutex_lock(&stratum_sock_lock);

			int validShares = pUCP->getValidShares();
			int invalidShares = pUCP->getInvalidShares();
			int totalAccountedForShares = invalidShares + validShares;
			int totalSubmittedShares = pUCP->getSentShares();
			int unaccountedForShares = totalSubmittedShares - totalAccountedForShares;

			pthread_mutex_unlock(&stratum_sock_lock);

			double percentage = ((double)validShares) / totalAccountedForShares;
			percentage *= 100;
			// printf("[GPU #%d (%s)] : %f MH/second    valid shares: %d/%d/%d (%.3f%%)\n", deviceToUse, selectedDeviceName.c_str(), hashSpeed, validShares, totalAccountedForShares, totalSubmittedShares, percentage);
			//printf("hashend %#018llx \n ", header[0]);
			printf("[GPU #%d (%s)] : %0.2f MH/s shares: %d/%d/%d (%.3f%%)\n", thr_id, device_name[thr_id], hashSpeed, validShares, totalAccountedForShares, totalSubmittedShares, percentage);


		}

		if (nonceResult[0] != 0x01000000 && nonceResult[0] != 0
			&& lastnonce[0] != nonceResult[0] && lastnonce[1] != nonceResult[0] && lastnonce[2] != nonceResult[0]
			&& lastnonce[3] != nonceResult[0]) {

			uint32_t nonce = *nonceResult;
			nonce = (((nonce & 0xFF000000) >> 24) | ((nonce & 0x00FF0000) >> 8) | ((nonce & 0x0000FF00) << 8) | ((nonce & 0x000000FF) << 24));
			pthread_mutex_lock(&stratum_sock_lock);
			lastnonce[3] = lastnonce[2];
			lastnonce[2] = lastnonce[1];
			lastnonce[1] = lastnonce[0];
			lastnonce[0] = nonceResult[0];


			pUCP->submitWork(jobId, timestamp, nonce);
			pthread_mutex_unlock(&stratum_sock_lock);
			nonceResult[0] = 0;

			char line[100];

			// Hash coming from GPU is reversed
			uint64_t hashFlipped = 0;
			hashFlipped |= (hashStart[0] & 0x00000000000000FF) << 56;
			hashFlipped |= (hashStart[0] & 0x000000000000FF00) << 40;
			hashFlipped |= (hashStart[0] & 0x0000000000FF0000) << 24;
			hashFlipped |= (hashStart[0] & 0x00000000FF000000) << 8;
			hashFlipped |= (hashStart[0] & 0x000000FF00000000) >> 8;
			hashFlipped |= (hashStart[0] & 0x0000FF0000000000) >> 24;
			hashFlipped |= (hashStart[0] & 0x00FF000000000000) >> 40;
			hashFlipped |= (hashStart[0] & 0xFF00000000000000) >> 56;

#if CPU_SHARES 
			sprintf(line, "\t Share Found @ 2^24! {%#018llx} [nonce: %#08lx]", hashFlipped, nonce);
#else
			sprintf(line, "\t Share Found @ 2^32! {%#08lx} [nonce: %#08lx]", hashFlipped, nonce);
#endif

			cout << line << endl;
			vprintf("Logging\n");
			Log::info(line);
			vprintf("Done logging\n");
			vprintf("Made line\n");

			numLines++;

			// Uncomment these lines to get access to this data for display purposes
			/*
			long long extraNonce = ucpClient.getStartExtraNonce();
			int jobId = ucpClient.getJobId();
			int encodedDifficulty = ucpClient.getEncodedDifficulty();
			string previousBlockHashHex = ucpClient.getPreviousBlockHash();
			string merkleRoot = ucpClient.getMerkleRoot();
			*/

		}
		vprintf("About to restart loop...\n");
	}

	printf("Resetting device...\n");
	clReleaseMemObject(pnoncestart_d);
	clReleaseMemObject(pnonceout_d);
	clReleaseMemObject(phashstartout_d);
	clReleaseMemObject(pheaderin_d);
	//	free(pnoncestart);
	//free(pnonceout);
	//free(phashstartout);
	//free(pheaderin);
	getchar();
	return 0;


}




int main(int argc, char *argv[])
{
	// Check for help argument (only -h)

	for (int i = 1; i < argc; i++) {
		char* argument = argv[i];

		if (!strcmp(argument, "-h"))
		{
			printHelpAndExit();
		}
	}

	if (argc % 2 != 1) {
		sprintf(outputBuffer, "GPU miner must be provided valid argument pairs!");
		cerr << outputBuffer << endl;
		printHelpAndExit();
	}

	string hostAndPort = ""; //  "94.130.64.18:8501";
	string username = ""; // "VGX71bcRsEh4HZzhbA9Nj7GQNH5jGw";
	string password = "";

	if (argc > 1)
	{
		for (int i = 1; i < argc; i += 2)
		{
			char* argument = argv[i];
			printf("%s\n", argument);
			if (argument[0] == '-' && argument[1] == 'd')
			{

				int device_thr[MAX_GPUS] = { 0 };
				cl_uint ngpus = clDevicesNum();
				char* pch = strtok(argv[i + 1], ",");
				opt_n_threads = 0;
				while (pch != NULL && opt_n_threads < MAX_GPUS) {
					if (pch[0] >= '0' && pch[0] <= '9' && strlen(pch) <= 2)
					{
						if (atoi(pch) < ngpus)
							device_map[opt_n_threads++] = atoi(pch);
						else {
							printf("Non-existant device #%d specified in -d option\n\n", atoi(pch));
							printHelpAndExit();
						}
					}
					pch = strtok(NULL, ",");
				}
				// count threads per gpu
				for (int n = 0; n < opt_n_threads; n++) {
					int device = device_map[n];
					device_thr[device]++;
				}
				for (int n = 0; n < ngpus; n++) {
					gpu_threads = max(gpu_threads, device_thr[n]);
				}

				//  if (strlen(argv[i + 1]) == 2) {
				//  device num >= 10
				//	deviceToUse = (argv[i + 1][0] - 48) * 10 + (argv[i + 1][1] - 48);
				//  }
				//  else {
				//	deviceToUse = argv[i + 1][0] - 48;
				//  }
			}
			else if (!strcmp(argument, "-o"))
			{
				hostAndPort = string(argv[i + 1]);
			}
			else if (!strcmp(argument, "-u"))
			{
				username = string(argv[i + 1]);
			}
			else if (!strcmp(argument, "-p"))
			{
				password = string(argv[i + 1]);
			}
			else if (!strcmp(argument, "-tpb"))
			{
				threadsPerBlock = stoi(argv[i + 1]);
			}
			else if (!strcmp(argument, "-bs"))
			{
				blocksize = stoi(argv[i + 1]);
			}
			else if (!strcmp(argument, "-frm"))
			{
				blocksize = stoi(argv[i + 1]);
			}
			else if (!strcmp(argument, "-l"))
			{
				// to lower case conversion
				for (int j = 0; j < strlen(argv[i + 1]); j++)
				{
					argv[i + 1][j] = tolower(argv[i + 1][j]);
				}
				if (!strcmp(argv[i + 1], "true") || !strcmp(argv[i + 1], "t"))
				{
					Log::setEnabled(true);
				}
				else
				{
					Log::setEnabled(false);
				}
			}
			else if (!strcmp(argument, "-v"))
			{
				// to lower case conversion
				for (int j = 0; j < strlen(argv[i + 1]); j++)
				{
					argv[i + 1][j] = tolower(argv[i + 1][j]);
				}
				if (!strcmp(argv[i + 1], "true") || !strcmp(argv[i + 1], "t"))
				{
					verboseOutput = true;
				}
				else
				{
					verboseOutput = false;
				}
			}
		}
	}
	else {
		printHelpAndExit();
	}
	pthread_mutex_init(&stratum_sock_lock, NULL);
	pthread_mutex_init(&stratum_log_lock, NULL);

	if (HIGH_RESOURCE) {
		sprintf(outputBuffer, "Resource Utilization: HIGH");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}
	else {
		sprintf(outputBuffer, "Resource Utilization: LOW");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}


#ifdef _WIN32
	HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
#else
#endif

	if (hostAndPort.compare("") == 0) {
		string error = "You must specify a host in the command line arguments! Example: \n-o 127.0.0.1:8501 or localhost:8501";
		cerr << error << endl;
		Log::error(error);
		promptExit(-1);
	}

	if (username.compare("") == 0) {
		string error = "You must specify a username in the command line arguments! Example: \n-u V5bLSbCqj9VzQR3MNANqL13YC2tUep";
		cerr << error << endl;
		Log::error(error);
		promptExit(-1);
	}

	string host = hostAndPort.substr(0, hostAndPort.find(":"));
	//GetHostByName
	net_init();
	host = net_dns_resolve(host.c_str());
	net_deinit();

	string portString = hostAndPort.substr(hostAndPort.find(":") + 1);

	// Ensure that port is numeric
	if (portString.find_first_not_of("1234567890") != string::npos) {
		string error = "You must specify a host in the command line arguments! Example: \n-o 127.0.0.1:8501 or localhost:8501";
		cerr << error << endl;
		Log::error(error);
		promptExit(-1);
	}

	int port = stoi(portString);

	sprintf(outputBuffer, "Attempting to mine to pool %s:%d with username %s and password %s...", host.c_str(), port, username.c_str(), password.c_str());
	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	UCPClient ucpClient(host, port, username, password);

	pUCP = &ucpClient;

	sprintf(outputBuffer, "Using Device: %d\n\n", deviceToUse);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);

	int version, ret;

	////////////////////////////////////////////
	cl_uint		nr_devs_total = opt_n_threads;
	cl_int		status;

	//scan_platforms(&plat_id, &dev_id);

	for (int i = 0; i < opt_n_threads; i++)
	{
		//	if (scan_platform(platform_M, &nr_devs_total, &plat_id, &dev_id[i], device_map[i]))

		/* Create context.*/
		context[i] = clCreateContext(NULL, 1, &dev_id[i],
			NULL, NULL, &status);
		if (status != CL_SUCCESS || !context[i]) {
			sprintf(outputBuffer, "clCreateContext (%d)", status);
			cout << outputBuffer << endl;
			Log::error(outputBuffer);
			promptExit(-1);
		}
		/* Creating command queue associate with the context.*/
		queue[i] = clCreateCommandQueue(context[i], dev_id[i],
			0, &status);
		if (status != CL_SUCCESS || !queue[i]) {
			sprintf(outputBuffer, "clCreateCommandQueue (%d)", status);
			cout << outputBuffer << endl;
			Log::error(outputBuffer);
			promptExit(-1);
		}

		/* Create program object */
#ifdef WIN32
		load_file("input.cl", (char **)&source, &source_len, 0);
		load_file("input.bin", &binary, &binary_len, 1);
#else
		source = ocl_code;
#endif
		source_len = strlen(source);

		program[i] = clCreateProgramWithSource(context[i], 1, (const char **)&source,
			&source_len, &status);
		if (status != CL_SUCCESS || !program[i]) {
			sprintf(outputBuffer, "clCreateProgramWithSource (%d)", status);
			cout << outputBuffer << endl;
			Log::error(outputBuffer);
			promptExit(-1);
		}

		/* Build program. */
		sprintf(outputBuffer, "Building program #%d", i);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);

		status = clBuildProgram(program[i], 1, &dev_id[i],
			(amd_flag) ? ("-I .. -I .") : ("-I .. -I ."), // compile options
			NULL, NULL);
		if (status != CL_SUCCESS) {
			sprintf(outputBuffer, "OpenCL build failed (%d). Build log follows:", status);

			get_program_build_log(program[i], dev_id[i]);
			cout << outputBuffer << endl;
			Log::error(outputBuffer);
			promptExit(-1);
		}

		get_program_bins(program[i]);
		// Create kernel objects
		k_vblake[i] = clCreateKernel(program[i], "kernel_vblake", &status);
		if (status != CL_SUCCESS || !k_vblake[i]) {
			sprintf(outputBuffer, "clCreateKernel (%d)", status);
			cout << outputBuffer << endl;
			Log::error(outputBuffer);
			get_program_build_log(program[i], dev_id[i]);
			promptExit(-1);
		}

	}

	pthread_t tids[MAX_GPUS];
	struct mining_attr m_args[MAX_GPUS];

	for (int i = 0; i < opt_n_threads; i++) {

		m_args[i].dev_id = device_map[i];

		pthread_create(&tids[i], NULL, miner_thread, &m_args[i]);
	}

	for (int i = 0; i < opt_n_threads; i++) {
		pthread_join(tids[i], NULL);
	}


}