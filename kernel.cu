#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <ctime>
#include <stdio.h>

#include <stdlib.h>
#include <ctime>
#include <sstream>

#include "Log.h"
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

cudaStream_t cudastream;

uint32_t *blockHeadermobj = nullptr;
uint32_t *midStatemobj = nullptr;
uint32_t *nonceOutmobj = nullptr;

cudaError_t grindNonces(uint32_t *dev_nonceStart, uint64_t* dev_header, uint32_t* dev_nonceResult,
                        uint64_t* dev_hashStart, uint32_t *nonceResult, uint64_t *hashStart, const
                        uint64_t *header, int deviceIndex, int threadsPerBlock, int blockSize);


__device__ __forceinline__
uint2 ROR2(const uint2 a, const int offset)
{
	uint2 result;
#if __CUDA_ARCH__ > 300
	if (offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else /* if (offset < 64) */ {
		/* offset SHOULD BE < 64 ! */
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
#else
	if (!offset)
		result = a;
	else if (offset < 32) {
		result.y = ((a.y >> offset) | (a.x << (32 - offset)));
		result.x = ((a.x >> offset) | (a.y << (32 - offset)));
	}
	else if (offset == 32) {
		result.y = a.x;
		result.x = a.y;
	}
	else {
		result.y = ((a.x >> (offset - 32)) | (a.y << (64 - offset)));
		result.x = ((a.y >> (offset - 32)) | (a.x << (64 - offset)));
	}
#endif
	return result;
}
static __device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }
#define B2B_G(v,a,b,c,d,x,y,c1,c2) { \
	v[a] = v[a] + v[b] + (x ^ c1); \
	v[d] ^= v[a]; \
	((uint2*)&v[d])[0] = ROR2(((uint2*)&v[d])[0], 60); \
	v[c] = v[c] + v[d]; \
	((uint2*)&v[b])[0] = ROR2(((uint2*)&v[b])[0] ^ ((uint2*)&v[c])[0], 43); \
	v[a] = v[a] + v[b] + (y ^ c2); \
	((uint2*)&v[d])[0] = ROR2(((uint2*)&v[d])[0] ^ ((uint2*)&v[a])[0], 5); \
	v[c] = v[c] + v[d]; \
	((uint2*)&v[b])[0] = ROR2(((uint2*)&v[b])[0] ^ ((uint2*)&v[c])[0], 18); \
	v[d] ^= ~(v[a] | v[b] | v[c]) | (~v[a] & v[b] & v[c]) | (v[a] & ~v[b] & v[c])   | (v[a] & v[b] & ~v[c]); \
    v[d] ^= (~v[a] & ~v[b] & v[c]) | (~v[a] & v[b] & ~v[c]) | (v[a] & ~v[b] & ~v[c]) | (v[a] & v[b] & v[c]); \
}


cudaError_t grindNonces(uint32_t *nonceResult, uint64_t *hashStart, const uint64_t *header);
static __constant__ const int8_t sigma[16][16] = {
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
static const uint64_t u512[16] =
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

__global__ void vblakeHasher(const uint32_t *nonceStart, uint32_t *nonceOut, uint64_t *hashStartOut, uint64_t * __restrict__ headerIn)
{
	const uint64_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) + nonceStart[0];
	uint64_t m[16] = { 0,0,0,0, 0,0,0,0 ,0,0,0,0, 0,0,0,0 };
	m[0] = headerIn[0];
	m[1] = headerIn[1];
	m[2] = headerIn[2];
	m[3] = headerIn[3];
	m[4] = headerIn[4];
	m[5] = headerIn[5];
	m[6] = headerIn[6];
	m[7] = headerIn[7];
	((uint32_t*)&m[0])[15] = nonce;

	uint64_t v[16] = {
		0x4bbf42c1f107ad85ull, 0x5D11A8C3B5AEB12Eull,
		0xA64AB78DC2774652ull, 0xC67595724658F253ull,
		0xB8864E79CB891E56ull, 0x12ED593E29FB41A1ull,
		0xB1DA3AB63C60BAA8ull, 0x6D20E50C1F954DEDull,
		0x4BBF42C1F006AD9Dull, 0x5D11A8C3B5AEB12Eull,
		0xA64AB78DC2774652ull, 0xC67595724658F253ull,
		0xb8864e79cb891e16ull, 0x12ED593E29FB41A1ull,
		0x4e25c549c39f4557ull, 0x6D20E50C1F954DEDull
	};

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
	uint64_t h64 = 0x3C10ED058B3FE57E ^ v[0] ^ v[8] ^ v[3] ^ v[11] ^ v[6] ^ v[14];

	if ((h64 & 0x00000000FFFFFFFFu) == 0) {

		nonceOut[0] = nonce;
	}
}

#define SHARE_SUBMISSION_NO_RESPONSE_WARN_THRESHOLD 50

uint32_t lastNonceStart = 0;

// Grind Through vBlake nonces with the provided header, setting the resultant nonce and associated hash start if a high-difficulty solution is found
cudaError_t grindNonces(uint32_t *dev_nonceStart, uint64_t* dev_header, uint32_t* dev_nonceResult,
                        uint64_t* dev_hashStart, uint32_t *nonceResult, uint64_t *hashStart, const
                        uint64_t *header, int deviceIndex, int threadsPerBlock, int blockSize)
{
	// Ensure that nonces don't overlap previous work
	uint32_t nonceStart = (uint64_t)lastNonceStart + (WORK_PER_THREAD * blockSize * threadsPerBlock);
	lastNonceStart = nonceStart;

	cudaError_t cudaStatus;

	// Copy starting nonce to GPU
	cudaStatus = cudaMemcpy(dev_nonceStart, &nonceStart, sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_header, header, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Zero out hash and nonce result
	cudaStatus = cudaMemset(dev_hashStart, 0, 1 * sizeof(uint64_t));
	cudaStatus = cudaMemset(dev_nonceResult, 0, 1 * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemset failed!");
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}
	blockSize *= WORK_PER_THREAD;
	// Launch a kernel on the GPU with one thread for each element.
	vblakeHasher << < blockSize, threadsPerBlock >> >(dev_nonceStart, dev_nonceResult, dev_hashStart, dev_header);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "grindNonces launch failed: %s\n", cudaGetErrorString(cudaStatus));
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaDeviceSynchronize returned error code %d after launching grindNonces!\n", cudaStatus);
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(nonceResult, dev_nonceResult, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hashStart, dev_hashStart, 1 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		sprintf(outputBuffer, "cudaMemcpy failed!");
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		cudaError_t e = cudaGetLastError();
		sprintf(outputBuffer, "Cuda Error: %s\n", cudaGetErrorString(e));
		std::cerr << outputBuffer << endl;
		Log::error(outputBuffer);
		goto Error;
	}

Error:
	return cudaStatus;
}