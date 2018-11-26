#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <ctime>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include "UCPClient.h"

#ifdef _WIN32
#include <Windows.h>
#include <VersionHelpers.h>
#elif __linux__
#include <sys/socket.h> 
#include <netdb.h>
#endif

#include <ctime>
#include "Log.h"
#include <sstream>
#include "Constants.h"

#if NVML
#include "nvml.h"
#endif

// #pragma comment(lib, "nvml.lib")
// #pragma comment(lib, "nvapi.lib")
// #pragma comment(lib, "nvapi64.lib")

void promptExit(int exitCode);
pthread_mutex_t stratum_sock_lock;
pthread_mutex_t stratum_log_lock;

#ifdef __INTELLISENSE__
#define __launch_bounds__(blocksize)
#endif
//#define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#define ROTR(x,n) ROTR64(x,n)
#define MAX_GPUS 16
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		fprintf(stderr, "Cuda error in func '%s' at line %i : %s.\n", \
		         __FUNCTION__, __LINE__, cudaGetErrorString(err) );   \
		promptExit(-1);                                           \
	}                                                                 \
} while (0)



__constant__ static uint64_t __align__(8) c_512[16];
__constant__ static uint64_t __align__(8) c_vblake[8];
__constant__ uint64_t headerIn[8];
static uint32_t *d_nonces[MAX_GPUS];
static uint64_t *dev_nonceStart[MAX_GPUS];


__host__
void veri_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_nonces[thr_id], 1 * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc(&dev_nonceStart[thr_id], 1 * sizeof(uint64_t)));
};
void veri_setBlock(void *blockf)
{
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(headerIn, blockf, 8 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
};

__device__ __forceinline__ uint64_t ROTR64_L(uint64_t value,
	const int offset) {
	uint2 result;
			asm("shf.r.wrap.b32 %0, %1, %2, %3;" :
		"=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))),
			"r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" :
		"=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))),
			"r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	
	return __double_as_longlong(__hiloint2double(result.y,
		result.x));
}
__device__ __forceinline__ uint64_t ROTR64_H(uint64_t value,
	const int offset) {
	uint2 result;

		asm("shf.r.wrap.b32 %0, %1, %2, %3;" :
		"=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))),
			"r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" :
		"=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))),
			"r"(__double2hiint(__longlong_as_double(value))), "r"(offset));

	return __double_as_longlong(__hiloint2double(result.y,
		result.x));
}
#define B2B_G(v,a,b,c,d,x,y,c1,c2) { \
	v[a] = v[a] + v[b] + (x ^ c1); \
	v[d] ^= v[a]; \
	v[d] = ROTR64_H(v[d], 60); \
	v[c] = v[c] + v[d]; \
	v[b] = ROTR64_H(v[b] ^ v[c], 43); \
	v[a] = v[a] + v[b] + (y ^ c2); \
	v[d] = ROTR64_L(v[d] ^ v[a], 5); \
	v[c] = v[c] + v[d]; \
	v[b] = ROTR64_L(v[b] ^ v[c], 18); \
	v[d] ^= (~v[a] & ~v[b] & ~v[c]) | (~v[a] & v[b] & v[c]) | (v[a] & ~v[b] & v[c])   | (v[a] & v[b] & ~v[c]); \
    v[d] ^= (~v[a] & ~v[b] & v[c]) | (~v[a] & v[b] & ~v[c]) | (v[a] & ~v[b] & ~v[c]) | (v[a] & v[b] & v[c]); \
}


void grindNonces(uint32_t startNonce, uint32_t *nonceResult, uint64_t *hashStart, const uint64_t *header, int dev_id);
__device__ __constant__
static const uint8_t c_sigma_big[16][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },

	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 },

	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};

static const uint64_t cpu_u512[16] =
{
	0xA51B6A89D489E800ULL, 0xD35B2E0E0B723800ULL,
	0xA47B39A2AE9F9000ULL, 0x0C0EFA33E77E6488ULL,
	0x4F452FEC309911EBULL, 0x3CFCC66F74E1022CULL,
	0x4606AD364DC879DDULL, 0xBBA055B53D47C800ULL,
	0x531655D90C59EB1BULL, 0xD1A00BA6DAE5B800ULL,
	0x2FE452DA9632463EULL, 0x98A7B5496226F800ULL,
	0xBAFCD004F92CA000ULL, 0x64A39957839525E7ULL,
	0xD859E6F081AAE000ULL, 0x63D980597B560E6BULL
};

static const uint64_t cpu_vBlake_iv[8] = {
	0x4BBF42C1F006AD9Dull, 0x5D11A8C3B5AEB12Eull,
	0xA64AB78DC2774652ull, 0xC67595724658F253ull,
	0xB8864E79CB891E56ull, 0x12ED593E29FB41A1ull,
	0xB1DA3AB63C60BAA8ull, 0x6D20E50C1F954DEDull
};


__device__ 
uint64_t vBlake2(const uint64_t h0, const uint64_t h1, const uint64_t h2, const uint64_t h3, const uint64_t h4, const uint64_t h5, const uint64_t h6, const uint64_t h7, const uint64_t* u512, const uint64_t* s_vBlake, const uint8_t((*sigma)[16]))
{
	uint64_t h[8];
    uint64_t v[16];
	uint64_t m[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

	h[0]  = v[8]  =  s_vBlake[0];
	
	h[0] ^= (uint64_t)(0x01010000 ^ 0x18);
	v[0] = h[0];
	v[9]  = v[1] = s_vBlake[1];
	v[10] = v[2] = s_vBlake[2];
	h[3]  = v[11] = v[3] = s_vBlake[3];
	v[12] = v[4] = s_vBlake[4];
	v[13] = v[5] = s_vBlake[5];
	h[6]  = v[14] = v[6] = s_vBlake[6];
	v[15] = v[7] = s_vBlake[7];
	
	

	m[0] = h0;
	m[1] = h1;
	m[2] = h2;
	m[3] = h3;
	m[4] = h4;
	m[5] = h5;
	m[6] = h6;
	m[7] = h7;		
	//vblake512_compress(h, b, c_sigma_big, s_u512);
	v[12] ^= 64;
	v[14] ^= (uint64_t)(0xffffffffffffffffull);// (long)(-1);

	#pragma unroll 16
	for (int i = 0; i < 16; i++) {
		B2B_G(v, 0, 4, 8, 12, m[sigma[i][1]], m[sigma[i][0]],
			u512[sigma[i][1]], u512[sigma[i][0]]);

		B2B_G(v, 1, 5, 9, 13, m[sigma[i][3]], m[sigma[i][2]],
			u512[sigma[i][3]], u512[sigma[i][2]]);

		B2B_G(v, 2, 6, 10, 14, m[sigma[i][5]], m[sigma[i][4]],
			u512[sigma[i][5]], u512[sigma[i][4]]);

		B2B_G(v, 3, 7, 11, 15, m[sigma[i][7]], m[sigma[i][6]],
			u512[sigma[i][7]], u512[sigma[i][6]]);

		B2B_G(v, 0, 5, 10, 15, m[sigma[i][9]], m[sigma[i][8]],
			u512[sigma[i][9]], u512[sigma[i][8]]);

		B2B_G(v, 1, 6, 11, 12, m[sigma[i][11]], m[sigma[i][10]],
			u512[sigma[i][11]], u512[sigma[i][10]]);

		B2B_G(v, 2, 7, 8, 13, m[sigma[i][13]], m[sigma[i][12]],
			u512[sigma[i][13]], u512[sigma[i][12]]);

		B2B_G(v, 3, 4, 9, 14, m[sigma[i][15]], m[sigma[i][14]],
			u512[sigma[i][15]], u512[sigma[i][14]]);
	}

	h[0] ^= v[0] ^ v[8];

	h[3] ^= v[3] ^ v[11];

	h[6] ^= v[6] ^ v[14];

	h[0] ^= h[3] ^ h[6];  //copied from  the java
	return h[0];
}


#if CPU_SHARES
#define WORK_PER_THREAD 256
#else
#define WORK_PER_THREAD 1
#endif

#if HIGH_RESOURCE
#define DEFAULT_BLOCKSIZE 0x80000
#define DEFAULT_THREADS_PER_BLOCK 256
#else
#define DEFAULT_BLOCKSIZE 512
#define DEFAULT_THREADS_PER_BLOCK 512
#endif

int blocksize = DEFAULT_BLOCKSIZE;
int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
int opt_n_threads = 0;
short device_map[MAX_GPUS] = { 0 };
int gpu_threads = 1;
int active_gpus;
char * device_name[MAX_GPUS];
long  device_sm[MAX_GPUS] = { 0 };
short device_mpcount[MAX_GPUS] = { 0 };
int init[MAX_GPUS] = { 0 };

bool verboseOutput = false;
struct mining_attr {
	int dev_id;
	string host;
	int port;
	string username;
	string password;

};

/*
* Kernel function to search a range of nonces for a solution falling under the macro-configured difficulty (CPU=2^24, GPU=2^32).
*/
__global__ void vblakeHasher(uint32_t startnonce, uint32_t *nonceOut, uint64_t *hashStartOut)
{
	// Generate a unique starting nonce for each thread that doesn't overlap with the work of any other thread
	uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) + startnonce;
	__shared__ uint64_t s_u512[16],s_vblake[8];

	if (threadIdx.x < 16U) s_u512[threadIdx.x] = c_512[threadIdx.x];
	if (threadIdx.x < 8U) s_vblake[threadIdx.x] = c_vblake[threadIdx.x];

	uint64_t nonceHeaderSection = headerIn[7];
	
	//for (unsigned int nonce = workStart; nonce < workStart + WORK_PER_THREAD; nonce++) {
		// Zero out nonce position and write new nonce to last 32 bits of prototype header
		nonceHeaderSection &= 0x00000000FFFFFFFFu;
		nonceHeaderSection |= (((uint64_t)nonce) << 32);

		uint64_t hashStart = vBlake2(headerIn[0], headerIn[1], headerIn[2], headerIn[3], headerIn[4], headerIn[5], headerIn[6], nonceHeaderSection, s_u512, s_vblake, c_sigma_big);

		if ((hashStart & 0x00000000FFFFFFFFu) == 0) {
			// Check that found solution is better than existing solution if one has already been found on this run of the kernel (always send back highest-quality work)
			if (hashStartOut[0] > hashStart || hashStartOut[0] == 0) {
				nonceOut[0] = nonce;
				hashStartOut[0] = hashStart;
			}

			// exit loop early
			//nonce = workStart + WORK_PER_THREAD;
		}
	//}
}
int cuda_num_devices()
{
	int version = 0, GPU_N = 0;
	cudaError_t err = cudaDriverGetVersion(&version);
	if (err != cudaSuccess) {
		printf("Unable to query CUDA driver version! Is an nVidia driver installed?\n");
		exit(1);
	}

	if (version < CUDART_VERSION) {
		printf("Your system does not support CUDA %d.%d API!\n",
			CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
		exit(1);
	}

	err = cudaGetDeviceCount(&GPU_N);
	if (err != cudaSuccess) {
		printf("Unable to query number of CUDA devices! Is an nVidia driver installed?\n");
		exit(1);
	}
	return GPU_N;
}
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
void cuda_devicenames()
{
	cudaError_t err;
	int GPU_N;
	err = cudaGetDeviceCount(&GPU_N);
	if (err != cudaSuccess)
	{
		printf("Unable to query number of CUDA devices! Is an nVidia driver installed?");
		exit(1);
	}

	if (opt_n_threads)
		GPU_N = min(MAX_GPUS, opt_n_threads);
	for (int i = 0; i < GPU_N; i++)
	{
			int dev_id = device_map[i];
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);

		device_sm[dev_id] = (props.major * 100 + props.minor * 10);
		device_mpcount[dev_id] = (short)props.multiProcessorCount;

		if (device_name[dev_id]) {
			free(device_name[dev_id]);
			device_name[dev_id] = NULL;
		}

			device_name[dev_id] = strdup(props.name);
	}
}
void printHelpAndExit() {
	printf("VeriBlock vBlake GPU CUDA Miner v1.0\n");
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
static bool substringsearch(const char *haystack, const char *needle, int &match)
{
	int hlen = (int)strlen(haystack);
	int nlen = (int)strlen(needle);
	for (int i = 0; i < hlen; ++i)
	{
		if (haystack[i] == ' ') continue;
		int j = 0, x = 0;
		while (j < nlen)
		{
			if (haystack[i + x] == ' ') { ++x; continue; }
			if (needle[j] == ' ') { ++j; continue; }
			if (needle[j] == '#') return ++match == needle[j + 1] - '0';
			if (tolower(haystack[i + x]) != tolower(needle[j])) break;
			++j; ++x;
		}
		if (j == nlen) return true;
	}
	return false;
}
int cuda_finddevice(char *name)
{
	int num = cuda_num_devices();
	int match = 0;
	for (int i = 0; i < num; ++i)
	{
		cudaDeviceProp props;
		if (cudaGetDeviceProperties(&props, i) == cudaSuccess)
			if (substringsearch(props.name, name, match)) return i;
	}
	return -1;
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

char outputBuffer[8192];


void* miner_thread(void* arg){
	// Run initialization of device before beginning timer
	struct mining_attr *arg_Struct =
		(struct mining_attr*) arg;

	pthread_mutex_lock(&stratum_sock_lock);
	UCPClient ucpClient(arg_Struct->host, arg_Struct->port, arg_Struct->username, arg_Struct->password);

	byte target[24];
	ucpClient.copyMiningTarget(target);
	uint64_t header[8];
	
	getWork(ucpClient, (uint32_t)std::time(0),header);
	pthread_mutex_unlock(&stratum_sock_lock);

	pthread_mutex_lock(&stratum_log_lock);
	unsigned long long startTime = std::time(0);
	pthread_mutex_unlock(&stratum_log_lock);
	//mutex unlock
	
	uint32_t nonceResult[1] = { 0 };
	uint64_t hashStart[1] = { 0 };
	uint32_t startNonce = 0;
	unsigned long long hashes = 0;
	uint32_t count = 0;
	int numLines = 0;

	// Mining loop
	while (true) {
		vprintf("top of mining loop\n");
		count++;
		long timestamp = (long)std::time(0);
		//delete[] header;
		vprintf("Getting work...\n");

		pthread_mutex_lock(&stratum_sock_lock);
		getWork(ucpClient, timestamp, header);
		vprintf("Getting job id...\n");
		int jobId = ucpClient.getJobId();
		pthread_mutex_unlock(&stratum_sock_lock);

		count++;
		vprintf("Running kernel...\n");
		grindNonces(startNonce, nonceResult, hashStart, header, arg_Struct->dev_id);
		
		vprintf("Kernel finished...\n");
		
		//mutex lock
		pthread_mutex_lock(&stratum_log_lock);
		unsigned long long totalTime = std::time(0) - startTime;
		pthread_mutex_unlock(&stratum_log_lock);
		//todo mutex unlock
		hashes += (blocksize * threadsPerBlock * WORK_PER_THREAD);
		if ((uint64_t)startNonce +  (uint64_t)(blocksize * threadsPerBlock * WORK_PER_THREAD) < (uint64_t)0xffffffff) {
			startNonce += (blocksize * threadsPerBlock * WORK_PER_THREAD);
		}
		else
			startNonce = 0;

		double hashSpeed = (double)hashes;
		hashSpeed /= (totalTime * 1024 * 1024);

		if (count % 10 == 0) {
			//mutex lock
			pthread_mutex_lock(&stratum_sock_lock);

			int validShares = ucpClient.getValidShares();
			int invalidShares = ucpClient.getInvalidShares();
			int totalAccountedForShares = invalidShares + validShares;
			int totalSubmittedShares = ucpClient.getSentShares();
			int unaccountedForShares = totalSubmittedShares - totalAccountedForShares;
			pthread_mutex_unlock(&stratum_sock_lock);
			//mutex unlock
			double percentage = ((double)validShares) / totalAccountedForShares;
			percentage *= 100;
			// printf("[GPU #%d (%s)] : %f MH/second    valid shares: %d/%d/%d (%.3f%%)\n", deviceToUse, selectedDeviceName.c_str(), hashSpeed, validShares, totalAccountedForShares, totalSubmittedShares, percentage);

			printf("[GPU: %d %s] : %0.2f MH/s shares: %d/%d/%d (%.3f%%)\n", arg_Struct->dev_id, device_name[arg_Struct->dev_id], hashSpeed, validShares, totalAccountedForShares, totalSubmittedShares, percentage);
		}

		if (nonceResult[0] != 0x01000000 && nonceResult[0] != 0) {
			uint32_t nonce = *nonceResult;
			nonce = (((nonce & 0xFF000000) >> 24) | ((nonce & 0x00FF0000) >> 8) | ((nonce & 0x0000FF00) << 8) | ((nonce & 0x000000FF) << 24));
			
			pthread_mutex_lock(&stratum_sock_lock);
			ucpClient.submitWork(jobId, timestamp, nonce);
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
			sprintf(line, "\t Share Found @ 2^32! {%#018llx} [nonce: %#08lx]", hashFlipped, nonce);
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
	CUDA_SAFE_CALL(cudaDeviceReset());
	
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
				int ngpus = cuda_num_devices();
				char* pch = strtok(argv[i + 1], ",");
				opt_n_threads = 0;
				while (pch != NULL && opt_n_threads < MAX_GPUS) {
					if (pch[0] >= '0' && pch[0] <= '9' && strlen(pch) <= 2)
					{
						if (atoi(pch) < ngpus)
							device_map[opt_n_threads++] = atoi(pch);
						else {
							printf("Non-existant CUDA device #%d specified in -d option\n\n", atoi(pch));
							printHelpAndExit();
						}
					}
					else {
						int device = cuda_finddevice(pch);
						if (device >= 0 && device < ngpus)
							device_map[opt_n_threads++] = device;
						else {
							printf("Non-existant CUDA device '%s' specified in -d option\n\n", pch);
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


	if (CPU_SHARES) {
		sprintf(outputBuffer, "Share Type: CPU");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}
	else {
		sprintf(outputBuffer, "Share Type: GPU");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}

	if (BENCHMARK) {
		sprintf(outputBuffer, "Benchmark Mode: ENABLED");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}
	else {
		sprintf(outputBuffer, "Benchmark Mode: DISABLED");
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




	active_gpus = cuda_num_devices();
	if (active_gpus == 0) {
		printf("No CUDA devices found! terminating.\n");
		exit(1);
	}
	for (int i = 0; i < MAX_GPUS; i++) {
		device_map[i] = i % active_gpus;
		device_name[i] = NULL;
	}
	cuda_devicenames();


/*	for (int i = 0; i < opt_n_threads; i++) {
		cudaSetDevice(device_map[i]);
		cudaDeviceReset();
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaFuncSetCacheConfig(vblakeHasher, cudaFuncCachePreferL1);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Last error: %s\n", cudaGetErrorString(e));
		cout << outputBuffer << endl;
		Log::info(outputBuffer);

	}
*/

	pthread_t tids[MAX_GPUS];
	struct mining_attr m_args[MAX_GPUS];

	for (int i = 0; i < opt_n_threads; i++) {
		m_args[i].host = host;
		m_args[i].port = port;
		m_args[i].username = username;
		m_args[i].password = password;
		m_args[i].dev_id = device_map[i];

		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_create(&tids[i], &attr, miner_thread, &m_args[i]);
	}

	pthread_join(tids[0], NULL);

}
uint32_t lastNonceStart = 0;

// Grind Through vBlake nonces with the provided header, setting the resultant nonce and associated hash start if a high-difficulty solution is found
void grindNonces(uint32_t startnonce, uint32_t *nonceResult, uint64_t *hashStart, const uint64_t *header, int dev_id)
{
	// Select GPU to run on
	if (!init[dev_id])
	{
		CUDA_SAFE_CALL(cudaSetDevice(dev_id));
		cudaDeviceReset();
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaFuncSetCacheConfig(vblakeHasher, cudaFuncCachePreferL1);
	// Allocate GPU buffers for nonce result and header
		veri_init(dev_id);
		init[dev_id] = 1;
	}
	
	// Copy starting nonce to GPU
	
	veri_setBlock((void*)header);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_512, cpu_u512, sizeof(cpu_u512), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_vblake, cpu_vBlake_iv, sizeof(cpu_vBlake_iv), 0, cudaMemcpyHostToDevice));
	cudaMemset(d_nonces[dev_id], 0x00, 1 * sizeof(uint32_t));
	cudaMemset(dev_nonceStart[dev_id], 0x00, 1 * sizeof(uint64_t));

	// Launch a kernel on the GPU with one thread for each element.
	vblakeHasher << < blocksize, threadsPerBlock >> >(startnonce, d_nonces[dev_id], dev_nonceStart[dev_id]);
	cudaThreadSynchronize();
	cudaMemcpy(nonceResult, d_nonces[dev_id], 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(hashStart, dev_nonceStart[dev_id], 1 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
};
