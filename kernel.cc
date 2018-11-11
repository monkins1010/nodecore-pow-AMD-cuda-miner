#define NOMINMAX
#include <cstdint>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <CL/cl.h>
#include <chrono>
#include <ctime>
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


#define DEFAULT_BLOCKSIZE 0x1000
#define DEFAULT_THREADS_PER_BLOCK 128


int blocksize = DEFAULT_BLOCKSIZE;
int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;

bool verboseOutput = false;
int amd_flag = 0;
//typedef SSIZE_T ssize_t;
#define open _open
const char *source = NULL;
size_t source_len;
char *binary = NULL;
size_t binary_len;
uint32_t lastNonceStart = 0;
char outputBuffer[100 * 1024];
string selectedDeviceName;
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
#ifdef _WIN32
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
#endif
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
#ifdef _WIN32
	dump("dump.co", p, sizes);
#endif
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
	selectedDeviceName = name;
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
  selectedDeviceName = name;
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
unsigned scan_platform(cl_platform_id plat, cl_uint *nr_devs_total,
	cl_platform_id *plat_id, cl_device_id *dev_id)
{
	cl_device_type	typ = CL_DEVICE_TYPE_ALL;
	cl_uint		nr_devs = 0;
	cl_device_id	*devices;
	cl_int		status;
	unsigned		found = 0;
	unsigned		i;
	print_platform_info(plat);
	status = clGetDeviceIDs(plat, typ, 0, NULL, &nr_devs);

	if (status != CL_SUCCESS)
	{
		sprintf(outputBuffer, "clGetDeviceInfo (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	if (nr_devs == 0)
		return 0;

	devices = (cl_device_id *)malloc(nr_devs * sizeof(*devices));
	status = clGetDeviceIDs(plat, typ, nr_devs, devices, NULL);
	if (status != CL_SUCCESS)
	{
		sprintf(outputBuffer, "clGetDeviceInfo (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}
	i = 0;
	while (i < nr_devs)
	{
		print_device_info(*nr_devs_total, devices[i]);
		if (*nr_devs_total == deviceToUse)
		{
			found = 1;
			*plat_id = plat;
			*dev_id = devices[i];
			break;
		}
		(*nr_devs_total)++;
		i++;
	}
	free(devices);
	return found;
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
void scan_platforms(cl_platform_id *plat_id, cl_device_id *dev_id)
{
	cl_uint		nr_platforms;
	cl_platform_id	*platforms;
	cl_uint		i, nr_devs_total;
	cl_int		status;
	status = clGetPlatformIDs(0, NULL, &nr_platforms);
	if (status != CL_SUCCESS)
	{
		sprintf(outputBuffer, "Cannot get OpenCL platforms (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}
	if (1)
	{
		sprintf(outputBuffer, "Found %d OpenCL platform(s)", nr_platforms);
		cout << outputBuffer << endl;
		Log::info(outputBuffer);

	}

	platforms = (cl_platform_id *)malloc(nr_platforms * sizeof(*platforms));
	if (!platforms)
	{
		sprintf(outputBuffer, "malloc: %s", strerror(errno));
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	status = clGetPlatformIDs(nr_platforms, platforms, NULL);
	if (status != CL_SUCCESS)
	{
		sprintf(outputBuffer, "clGetPlatformIDs (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	i = nr_devs_total = 0;
	while (i < nr_platforms)
	{
		if (scan_platform(platforms[i], &nr_devs_total, plat_id, dev_id))
			break;
		i++;
	}
	sprintf(outputBuffer, "Using GPU device ID %d", deviceToUse);
	cout << outputBuffer << endl;
	Log::error(outputBuffer);
	amd_flag = is_platform_amd(*plat_id);
	free(platforms);
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

	////////////////////////////////////////////
	cl_platform_id	plat_id = 0;
	cl_device_id	dev_id = 0;
	cl_kernel		k_vblake;
	cl_int		status;

	scan_platforms(&plat_id, &dev_id);

	if (!plat_id || !dev_id) {
		sprintf(outputBuffer, "Selected device (ID %d) not found", deviceToUse);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	/* Create context.*/
	cl_context context = clCreateContext(NULL, 1, &dev_id,
		NULL, NULL, &status);
	if (status != CL_SUCCESS || !context) {
		sprintf(outputBuffer, "clCreateContext (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}
	/* Creating command queue associate with the context.*/
	cl_command_queue queue = clCreateCommandQueue(context, dev_id,
		0, &status);
	if (status != CL_SUCCESS || !queue) {
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
	cl_program program;

	program = clCreateProgramWithSource(context, 1, (const char **)&source,
		&source_len, &status);
	if (status != CL_SUCCESS || !program) {
		sprintf(outputBuffer, "clCreateProgramWithSource (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	/* Build program. */
	sprintf(outputBuffer, "Building program");
	cout << outputBuffer << endl;
	Log::info(outputBuffer);

	status = clBuildProgram(program, 1, &dev_id,
		(amd_flag) ? ("-I .. -I .") : ("-I .. -I ."), // compile options
		NULL, NULL);
	if (status != CL_SUCCESS) {
		sprintf(outputBuffer, "OpenCL build failed (%d). Build log follows:", status);



		get_program_build_log(program, dev_id);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		promptExit(-1);
	}

	get_program_bins(program);
	// Create kernel objects
	k_vblake = clCreateKernel(program, "kernel_vblake", &status);
	if (status != CL_SUCCESS || !k_vblake) {
		sprintf(outputBuffer, "clCreateKernel (%d)", status);
		cout << outputBuffer << endl;
		Log::error(outputBuffer);
		get_program_build_log(program, dev_id);
		promptExit(-1);
	}


	/////////////////////////////////


	// Print out information about all available CUDA devices on system


	sprintf(outputBuffer, "Mining on device #%d...", deviceToUse);
	cout << outputBuffer << endl;
	Log::info(outputBuffer);



	// Run initialization of device before beginning timer
	uint64_t* header = getWork(ucpClient, (uint32_t)std::time(0));

	unsigned long long startTime = std::time(0);
	uint32_t nonceResult[1] = { 0 };
	uint64_t hashStart[1] = { 0 };
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
	pheaderin = (uint64_t *)malloc(sizeof(uint64_t) * 8);

	pnoncestart_d = check_clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * 1, NULL);
	pnonceout_d = check_clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * 1, NULL);
	phashstartout_d = check_clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint64_t) * 1, NULL);
	pheaderin_d = check_clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint64_t) * 8, NULL);

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
		for (int i = 0; i<8; i++)
			pheaderin[i] = header[i];
		pnonceout[0] = 0;
		clEnqueueWriteBuffer(queue, pnonceout_d, CL_TRUE, 0, sizeof(uint32_t) * 1, pnonceout, 0, NULL, NULL);
		phashstartout[0] = 0;
		clEnqueueWriteBuffer(queue, pheaderin_d, CL_TRUE, 0, sizeof(uint64_t) * 8, pheaderin, 0, NULL, NULL);


		uint32_t nonceStart = (uint64_t)lastNonceStart + (blocksize * threadsPerBlock * 128);
		lastNonceStart = nonceStart;
		pnoncestart[0] = nonceStart;
		clEnqueueWriteBuffer(queue, pnoncestart_d, CL_TRUE, 0, sizeof(uint32_t) * 1, pnoncestart, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, phashstartout_d, CL_TRUE, 0, sizeof(uint64_t) * 1, phashstartout, 0, NULL, NULL);

		check_clSetKernelArg(k_vblake, 0, &pnoncestart_d);
		check_clSetKernelArg(k_vblake, 1, &pnonceout_d);
		check_clSetKernelArg(k_vblake, 2, &phashstartout_d);
		check_clSetKernelArg(k_vblake, 3, &pheaderin_d);

		global_ws = (unsigned int)(blocksize * threadsPerBlock * 128);

		//start Opencl kernel
		check_clEnqueueNDRangeKernel(queue, k_vblake, 1, NULL,
			&global_ws, &local_work_size, 0, NULL, NULL);
		clFinish(queue);

		check_clEnqueueReadBuffer(queue, pnonceout_d,
			CL_TRUE,	// cl_bool	blocking_read
			0,		// size_t	offset
			sizeof(uint32_t) * 1,	// size_t	size
			pnonceout,	// void		*ptr
			0,		// cl_uint	num_events_in_wait_list
			NULL,	// cl_event	*event_wait_list
			NULL);	// cl_event	*event

		check_clEnqueueReadBuffer(queue, phashstartout_d,
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

		unsigned long long totalTime = std::time(0) - startTime;
		hashes += (threadsPerBlock * blocksize * 128);

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
	clReleaseMemObject(pnoncestart_d);
	clReleaseMemObject(pnonceout_d);
	clReleaseMemObject(phashstartout_d);
	clReleaseMemObject(pheaderin_d);
	free(pnoncestart);
	free(pnonceout);
	free(phashstartout);
	free(pheaderin);
	getchar();
	return 0;


}


/*

// Grind Through vBlake nonces with the provided header, setting the resultant nonce and associated hash start if a high-difficulty solution is found
cl_int grindNonces(uint32_t *nonceResult, uint64_t *hashStart, const uint64_t *header, cl_kernel k_vblake, cl_command_queue queue,
cl_context context, cl_device_id dev_id,cl_program program)
{
// Device memory
uint32_t *dev_nonceStart = 0;
uint64_t *dev_header = 0;
uint32_t *dev_nonceResult = 0;
uint64_t *dev_hashStart = 0;

// Ensure that nonces don't overlap previous work
uint32_t nonceStart = (uint64_t)lastNonceStart + (WORK_PER_THREAD * blocksize * threadsPerBlock);
lastNonceStart = nonceStart;

cl_int clStatus;



// Allocate GPU buffers for nonce result and header


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
*/