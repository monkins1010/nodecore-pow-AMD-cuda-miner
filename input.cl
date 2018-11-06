
#define DEFAULT_BLOCKSIZE 512
#define DEFAULT_THREADS_PER_BLOCK 256
#define WORK_PER_THREAD 1024

#define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#define ROTR(x,n) ROTR64(x,n)


#define cuda_swab64(x) \
		((ulong)((((ulong)(x) & 0xff00000000000000UL) >> 56) | \
			(((ulong)(x) & 0x00ff000000000000UL) >> 40) | \
			(((ulong)(x) & 0x0000ff0000000000UL) >> 24) | \
			(((ulong)(x) & 0x000000ff00000000UL) >>  8) | \
			(((ulong)(x) & 0x00000000ff000000UL) <<  8) | \
			(((ulong)(x) & 0x0000000000ff0000UL) << 24) | \
			(((ulong)(x) & 0x000000000000ff00UL) << 40) | \
			(((ulong)(x) & 0x00000000000000ffUL) << 56)))


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


__constant static const uchar sigma[16][16] = {
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

__constant static const unsigned long u512[16] =
{
	0xA51B6A89D489E800UL, 0xD35B2E0E0B723800UL,
	0xA47B39A2AE9F9000UL, 0x0C0EFA33E77E6488UL,
	0x4F452FEC309911EBUL, 0x3CFCC66F74E1022CUL,
	0x4606AD364DC879DDUL, 0xBBA055B53D47C800UL,
	0x531655D90C59EB1BUL, 0xD1A00BA6DAE5B800UL,
	0x2FE452DA9632463EUL, 0x98A7B5496226F800UL,
	0xBAFCD004F92CA000UL, 0x64A39957839525E7UL,
	0xD859E6F081AAE000UL, 0x63D980597B560E6BUL
};

__constant static const unsigned long vBlake_iv[8] = {
	0x4BBF42C1F006AD9Dul, 0x5D11A8C3B5AEB12Eul,
	0xA64AB78DC2774652ul, 0xC67595724658F253ul,
	0xB8864E79CB891E56ul, 0x12ED593E29FB41A1ul,
	0xB1DA3AB63C60BAA8ul, 0x6D20E50C1F954DEDul
};


void vblake512_compress(unsigned long *h, const unsigned long *block)
{
	unsigned long v[16];
	unsigned long m[16];

	//#pragma unroll 8
	for (int i = 0; i < 8; i++) {
		v[i] = h[i];
		v[i + 8] = vBlake_iv[i];
	}

	v[12] ^= 64;
	//v[13] ^= 0;
	v[14] ^= (ulong)(0xfffffffffffffffful);// (long)(-1);
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

unsigned long vBlake2(const ulong h0, const ulong h1, const ulong h2, const ulong h3, const ulong h4, const ulong h5, const ulong h6, const ulong h7)
{
	unsigned long b[8];
	unsigned long h[8];

	for (int i = 0; i < 8; i++) {
		h[i] = vBlake_iv[i];
	}
	h[0] ^= (ulong)(0x01010000 ^ 0x18);

	b[0] = h0;
	b[1] = h1;
	b[2] = h2;
	b[3] = h3;
	b[4] = h4;
	b[5] = h5;
	b[6] = h6;
	b[7] = h7;

	vblake512_compress(h, b);

	//for (int i = 0; i < 8; i++) {
	//	b[0] = cuda_swab64(h[0]);
	//}
	return h[0];
}

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_vblake(__global uint *nonceStart, __global uint *nonceOut, __global unsigned long *hashStartOut, __global unsigned long  *headerIn)
{
	// Generate a unique starting nonce for each thread that doesn't overlap with the work of any other thread
	uint workStart = (uint)(get_global_id(0)) + nonceStart[0];

	unsigned long nonceHeaderSection = headerIn[7];

	// Run the hash WORK_PER_THREAD times
	// for (unsigned int nonce = workStart; nonce < workStart + WORK_PER_THREAD; nonce++) {
		// Zero out nonce position and write new nonce to last 32 bits of prototype header
	   uint  nonce = workStart;
		nonceHeaderSection &= 0x00000000FFFFFFFFu;
		nonceHeaderSection |= (((unsigned long)nonce) << 32);

		unsigned long hashStart = vBlake2(headerIn[0], headerIn[1], headerIn[2], headerIn[3], headerIn[4], headerIn[5], headerIn[6], nonceHeaderSection);

		if ((hashStart & 0x00000000FFFFFFFFu) == 0) { // 2^32 difficulty
						
			// Check that found solution is better than existing solution if one has already been found on this run of the kernel (always send back highest-quality work)
			if (hashStartOut[0] > hashStart || hashStartOut[0] == 0) {
				nonceOut[0] = nonce;
				hashStartOut[0] = hashStart;
			}

			// exit loop early
			nonce = workStart + WORK_PER_THREAD;
		}
	// }
}
