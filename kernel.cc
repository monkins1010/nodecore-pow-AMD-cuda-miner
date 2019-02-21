#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <ctime>
#include <stdio.h>

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

#ifdef __INTELLISENSE__
#define __launch_bounds__(blocksize)
#endif
#define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#define ROTR(x,n) ROTR64(x,n)
#define ROTL64(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))

#define cuda_swab64(x) \
		((uint64_t)((((uint64_t)(x) & 0xff00000000000000ULL) >> 56) | \
			(((uint64_t)(x) & 0x00ff000000000000ULL) >> 40) | \
			(((uint64_t)(x) & 0x0000ff0000000000ULL) >> 24) | \
			(((uint64_t)(x) & 0x000000ff00000000ULL) >>  8) | \
			(((uint64_t)(x) & 0x00000000ff000000ULL) <<  8) | \
			(((uint64_t)(x) & 0x0000000000ff0000ULL) << 24) | \
			(((uint64_t)(x) & 0x000000000000ff00ULL) << 40) | \
			(((uint64_t)(x) & 0x00000000000000ffULL) << 56)))
__device__ __forceinline__
uint64_t SWAPDWORDS(uint64_t value)
{
#if __CUDA_ARCH__ >= 320
	uint2 temp;
	asm("mov.b64 {%0, %1}, %2; ": "=r"(temp.x), "=r"(temp.y) : "l"(value));
	asm("mov.b64 %0, {%1, %2}; ": "=l"(value) : "r"(temp.y), "r"(temp.x));
	return value;
#else
	return ROTL64(value, 32);
#endif
}

#define B2B_G(v,a,b,c,d,x,y,c1,c2) { \
	v[a] = v[a] + v[b] + (x ^ c1); \
	v[d] ^= v[a]; \
	v[d] = ROTR64(v[d], 60); \
	v[c] = v[c] + v[d]; \
	v[b] = ROTR64(v[b] ^ v[c], 43); \
	v[a] = v[a] + v[b] + (y ^ c2); \
	v[d] = ROTR64(v[d] ^ v[a], 5); \
	v[c] = v[c] + v[d]; \
	v[b] = ROTR64(v[b] ^ v[c], 18); \
	v[d] ^= (~v[a] & ~v[b] & ~v[c]) | (~v[a] & v[b] & v[c]) | (v[a] & ~v[b] & v[c])   | (v[a] & v[b] & ~v[c]); \
    v[d] ^= (~v[a] & ~v[b] & v[c]) | (~v[a] & v[b] & ~v[c]) | (v[a] & ~v[b] & ~v[c]) | (v[a] & v[b] & v[c]); \
}
cudaStream_t cudastream;

uint32_t *blockHeadermobj = nullptr;
uint32_t *midStatemobj = nullptr;
uint32_t *nonceOutmobj = nullptr;

cudaError_t grindNonces(uint32_t *nonceResult, uint64_t *hashStart, const uint64_t *header);
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

__device__ __constant__
static const uint64_t c_u512[16] =
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

__device__ __constant__
static const uint64_t vBlake_iv[8] = {
	0x4BBF42C1F006AD9Dull, 0x5D11A8C3B5AEB12Eull,
	0xA64AB78DC2774652ull, 0xC67595724658F253ull,
	0xB8864E79CB891E56ull, 0x12ED593E29FB41A1ull,
	0xB1DA3AB63C60BAA8ull, 0x6D20E50C1F954DEDull
};

__device__
void vblake512_compress(uint64_t *h, const uint64_t *block, const uint8_t((*sigma)[16]), const uint64_t *u512)
{
	uint64_t v[16];
	uint64_t m[16];

	//#pragma unroll 8
	for (int i = 0; i < 8; i++) {
		v[i] = h[i];
		v[i + 8] = vBlake_iv[i];
	}

	v[12] ^= 64;
	//v[13] ^= 0;
	v[14] ^= (uint64_t)(0xffffffffffffffffull);// (long)(-1);
											   //v[15] ^= 0;

											   //#pragma unroll 8
	for (int i = 0; i < 8; i++) {
		m[i] = block[i]; // cuda_swab64(block[i]); orORINGNAL BLAKE
	}


	//#pragma unroll 16
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
	//	h[1] ^= v[1] ^ v[9];
	//	h[2] ^= v[2] ^ v[10];
	h[3] ^= v[3] ^ v[11];
	//	h[4] ^= v[4] ^ v[12];
	//	h[5] ^= v[5] ^ v[13];
	h[6] ^= v[6] ^ v[14];
	//	h[7] ^= v[7] ^ v[15];

	h[0] ^= h[3] ^ h[6];  //copied from  the java
						  //h[1] ^= h[4] ^ h[7];
						  //h[2] ^= h[5];
}
__device__ __forceinline__
uint64_t vBlake2(const uint64_t h0, const uint64_t h1, const uint64_t h2, const uint64_t h3, const uint64_t h4, const uint64_t h5, const uint64_t h6, const uint64_t h7)
{
	uint64_t b[8];
	uint64_t h[8];

	for (int i = 0; i < 8; i++) {
		h[i] = vBlake_iv[i];
	}
	h[0] ^= (uint64_t)(0x01010000 ^ 0x18);

	b[0] = h0;
	b[1] = h1;
	b[2] = h2;
	b[3] = h3;
	b[4] = h4;
	b[5] = h5;
	b[6] = h6;
	b[7] = h7;

	vblake512_compress(h, b, c_sigma_big, c_u512);

	//for (int i = 0; i < 8; i++) {
	//	b[0] = cuda_swab64(h[0]);
	//}
	return h[0];
}


#if CPU_SHARES
#define WORK_PER_THREAD 256
#else
#define WORK_PER_THREAD 256
#endif

#if HIGH_RESOURCE
#define DEFAULT_BLOCKSIZE 512
#define DEFAULT_THREADS_PER_BLOCK 1024
#else
#define DEFAULT_BLOCKSIZE 512
#define DEFAULT_THREADS_PER_BLOCK 512
#endif

int blocksize = DEFAULT_BLOCKSIZE;
int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;

bool verboseOutput = false;

/*
* Kernel function to search a range of nonces for a solution falling under the macro-configured difficulty (CPU=2^24, GPU=2^32).
*/
__launch_bounds__(256, 2)
__global__ void vblakeHasher(uint32_t *nonceStart, uint32_t *nonceOut, uint64_t *hashStartOut, uint64_t const *headerIn)
{
	// Generate a unique starting nonce for each thread that doesn't overlap with the work of any other thread
	const uint32_t workStart = ((blockDim.x * blockIdx.x + threadIdx.x)  * WORK_PER_THREAD) + nonceStart[0];

	uint64_t nonceHeaderSection = headerIn[7];

	// Run the hash WORK_PER_THREAD times
	for (unsigned int nonce = workStart; nonce < workStart + WORK_PER_THREAD; nonce++) {
		// Zero out nonce position and write new nonce to last 32 bits of prototype header
		nonceHeaderSection &= 0x00000000FFFFFFFFu;
		nonceHeaderSection |= (((uint64_t)nonce) << 32);

		uint64_t hashStart = vBlake2(headerIn[0], headerIn[1], headerIn[2], headerIn[3], headerIn[4], headerIn[5], headerIn[6], nonceHeaderSection);

		if ((hashStart &

#if CPU_SHARES
			0x0000000000FFFFFFu // 2^24 difficulty
#else
			0x00000000FFFFFFFFu // 2^32 difficulty
#endif
			) == 0) {
			// Check that found solution is better than existing solution if one has already been found on this run of the kernel (always send back highest-quality work)
			if (hashStartOut[0] > hashStart || hashStartOut[0] == 0) {
				nonceOut[0] = nonce;
				hashStartOut[0] = hashStart;
			}

			// exit loop early
			nonce = workStart + WORK_PER_THREAD;
		}
	}
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
uint64_t* getWork(UCPClient& ucpClient, uint32_t timestamp)
{
	uint64_t *header = new uint64_t[8];
	ucpClient.copyHeaderToHash((byte *)header);
	embedTimestampInHeader((uint8_t*)header, timestamp);
	return header;
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
				if (strlen(argv[i + 1]) == 2) {
					// device num >= 10
					deviceToUse = (argv[i + 1][0] - 48) * 10 + (argv[i + 1][1] - 48);
				}
				else {
					deviceToUse = argv[i + 1][0] - 48;
				}
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

	if (NVML) {
		sprintf(outputBuffer, "NVML Status: ENABLED");
		cerr << outputBuffer << endl;
		Log::info(outputBuffer);
	}
	else {
		sprintf(outputBuffer, "NVML Status: DISABLED");
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

	// No effect if NVML is not enabled
	readyNVML(deviceToUse);

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

	byte target[24];
	ucpClient.copyMiningTarget(target);

	sprintf(outputBuffer, "Using Device: %d\n\n", deviceToUse);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);

	int version, ret;
	ret = cudaDriverGetVersion(&version);
	if (ret != cudaSuccess)
	{
		sprintf(outputBuffer, "Error when getting CUDA driver version: %d", ret);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	int runtimeVersion;
	ret = cudaRuntimeGetVersion(&runtimeVersion);
	if (ret != cudaSuccess)
	{
		sprintf(outputBuffer, "Error when getting CUDA runtime version: %d", ret);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}


	int deviceCount;
	ret = cudaGetDeviceCount(&deviceCount);
	if (ret != cudaSuccess)
	{
		sprintf(outputBuffer, "Error when getting CUDA device count: %d", ret);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	cudaDeviceProp deviceProp;

#if NVML
	char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
	nvmlSystemGetDriverVersion(driver, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
#else
	char driver[] = "???.?? (NVML NOT ENABLED)";
#endif

	sprintf(outputBuffer, "CUDA Version: %.1f", ((float)version / 1000));
	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	sprintf(outputBuffer, "CUDA Runtime Version: %d", runtimeVersion);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	sprintf(outputBuffer, "NVidia Driver Version: %s", driver);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);
	sprintf(outputBuffer, "CUDA Devices: %d", deviceCount);
	cout << outputBuffer << endl << endl;
	Log::info(outputBuffer);

	string selectedDeviceName;
	// Print out information about all available CUDA devices on system
	for (int count = 0; count < deviceCount; count++)
	{
		ret = cudaGetDeviceProperties(&deviceProp, count);
		if (ret != cudaSuccess)
		{
			sprintf(outputBuffer, "An error occurred while getting the CUDA device properties: %d", ret);
			cerr << outputBuffer << endl;
			Log::error(outputBuffer);
		}

		if (count == deviceToUse) {
			selectedDeviceName = deviceProp.name;
		}

		sprintf(outputBuffer, "Device #%d (%s):", count, deviceProp.name);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Clock Rate:              %d MHz", (deviceProp.clockRate / 1024));
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Is Integrated:           %s", (deviceProp.integrated == 0 ? "false" : "true"));
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Compute Capability:      %d.%d", deviceProp.major, deviceProp.minor);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Kernel Concurrency:      %d", deviceProp.concurrentKernels);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Max Grid Size:           %d x %d x %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Max Threads per Block:   %d", deviceProp.maxThreadsPerBlock);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Registers per Block:     %d", deviceProp.regsPerBlock);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Registers per SM:        %d", deviceProp.regsPerMultiprocessor);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Processor Count:         %d", deviceProp.multiProcessorCount);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Shared Memory/Block:     %zd", deviceProp.sharedMemPerBlock);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Shared Memory/Proc:      %zd", deviceProp.sharedMemPerMultiprocessor);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
		sprintf(outputBuffer, "    Warp Size:               %d", deviceProp.warpSize);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);
	}

	sprintf(outputBuffer, "Mining on device #%d...", deviceToUse);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);

	ret = cudaSetDevice(deviceToUse);
	if (ret != cudaSuccess)
	{
		sprintf(outputBuffer, "CUDA encountered an error while setting the device to %d:%d", deviceToUse, ret);
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
	}

	cudaDeviceReset();

	// Don't have GPU busy-wait on GPU
	ret = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

	cudaError_t e = cudaGetLastError();
	sprintf(outputBuffer, "Last error: %s\n", cudaGetErrorString(e));
	cout << outputBuffer << endl;
	Log::info(outputBuffer);

	// Run initialization of device before beginning timer
	uint64_t* header = getWork(ucpClient, (uint32_t)std::time(0));

	unsigned long long startTime = std::time(0);
	uint32_t nonceResult[1] = { 0 };
	uint64_t hashStart[1] = { 0 };

	unsigned long long hashes = 0;
	cudaError_t cudaStatus;

	uint32_t count = 0;

	int numLines = 0;

	// Mining loop
	while (true) {
		vprintf("top of mining loop\n");
		count++;
		long timestamp = (long)std::time(0);
		delete[] header;
		vprintf("Getting work...\n");
		header = getWork(ucpClient, timestamp);
		vprintf("Getting job id...\n");
		int jobId = ucpClient.getJobId();
		count++;
		vprintf("Running kernel...\n");
		cudaStatus = grindNonces(nonceResult, hashStart, header);
		vprintf("Kernel finished...\n");
		if (cudaStatus != cudaSuccess) {
			cudaError_t e = cudaGetLastError();
			sprintf(outputBuffer, "Error from running grindNonces: %s\nThis often occurs when a GPU overheats, has an unstable overclock, or has too aggressive launch parameters\nfor the vBlake kernel.\nYou can try using less aggressive settings, like:\n-tpb 256 -bs 256\nAnd try increasing these numbers until you hit instability issues again.", cudaGetErrorString(e));
			cerr << outputBuffer << endl;
			Log::error(outputBuffer);
			promptExit(-1);
		}

		unsigned long long totalTime = std::time(0) - startTime;
		hashes += (blocksize * threadsPerBlock * WORK_PER_THREAD);

		double hashSpeed = (double)hashes;
		hashSpeed /= (totalTime * 1024 * 1024);

		if (count % 10 == 0) {
			int validShares = ucpClient.getValidShares();
			int invalidShares = ucpClient.getInvalidShares();
			int totalAccountedForShares = invalidShares + validShares;
			int totalSubmittedShares = ucpClient.getSentShares();
			int unaccountedForShares = totalSubmittedShares - totalAccountedForShares;

			double percentage = ((double)validShares) / totalAccountedForShares;
			percentage *= 100;
			// printf("[GPU #%d (%s)] : %f MH/second    valid shares: %d/%d/%d (%.3f%%)\n", deviceToUse, selectedDeviceName.c_str(), hashSpeed, validShares, totalAccountedForShares, totalSubmittedShares, percentage);

			printf("[GPU #%d (%s)] : %0.2f MH/s shares: %d/%d/%d (%.3f%%)\n", deviceToUse, selectedDeviceName.c_str(), hashSpeed, validShares, totalAccountedForShares, totalSubmittedShares, percentage);
		}

		if (nonceResult[0] != 0x01000000 && nonceResult[0] != 0) {
			uint32_t nonce = *nonceResult;
			nonce = (((nonce & 0xFF000000) >> 24) | ((nonce & 0x00FF0000) >> 8) | ((nonce & 0x0000FF00) << 8) | ((nonce & 0x000000FF) << 24));

			ucpClient.submitWork(jobId, timestamp, nonce);

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
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	printf("Done resetting device...\n");

	getchar();
	return 0;
}

uint32_t lastNonceStart = 0;

// Grind Through vBlake nonces with the provided header, setting the resultant nonce and associated hash start if a high-difficulty solution is found
cudaError_t grindNonces(uint32_t *nonceResult, uint64_t *hashStart, const uint64_t *header)
{
	// Device memory
	uint32_t *dev_nonceStart = 0;
	uint64_t *dev_header = 0;
	uint32_t *dev_nonceResult = 0;
	uint64_t *dev_hashStart = 0;

	// Ensure that nonces don't overlap previous work
	uint32_t nonceStart = (uint64_t)lastNonceStart + (WORK_PER_THREAD * blocksize * threadsPerBlock);
	lastNonceStart = nonceStart;

	cudaError_t cudaStatus;

	// Select GPU to run on
	cudaStatus = cudaSetDevice(deviceToUse);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Allocate GPU buffers for nonce result and header
	cudaStatus = cudaMalloc((void**)&dev_nonceStart, 1 * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Copy starting nonce to GPU
	cudaStatus = cudaMemcpy(dev_nonceStart, &nonceStart, sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Allocate GPU buffers for nonce result and header.
	cudaStatus = cudaMalloc((void**)&dev_nonceResult, 1 * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Allocate GPU buffers for nonce result and header.
	cudaStatus = cudaMalloc((void**)&dev_hashStart, 1 * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_header, 8 * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_header, header, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMalloc failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	vblakeHasher << < blocksize, threadsPerBlock >> >(dev_nonceStart, dev_nonceResult, dev_hashStart, dev_header);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "grindNonces launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaDeviceSynchronize returned error code %d after launching grindNonces!\n", cudaStatus);
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(nonceResult, dev_nonceResult, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hashStart, dev_hashStart, 1 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

Error:
	cudaFree(dev_nonceStart);
	cudaFree(dev_header);
	cudaFree(dev_nonceResult);
	cudaFree(dev_hashStart);
	return cudaStatus;
}