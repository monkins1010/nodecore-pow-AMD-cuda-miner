#ifdef cl_amd_media_ops
#define ROTR64(x, n) ((n) < 32 ? (amd_bitalign((uint)((x) >> 32), (uint)(x), (uint)(n)) | ((ulong)amd_bitalign((uint)(x), (uint)((x) >> 32), (uint)(n)) << 32)) : (amd_bitalign((uint)(x), (uint)((x) >> 32), (uint)(n) - 32) | ((ulong)amd_bitalign((uint)((x) >> 32), (uint)(x), (uint)(n) - 32) << 32)))
#else
#define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#endif

#define B2B_G(v,a,b,c,d,x,y,c1,c2) { \
	v[a] += v[b] + (x ^ c1); \
	v[d] ^= v[a]; \
	v[d] = ROTR64(v[d], 60); \
	v[c] += v[d]; \
	v[b] = ROTR64(v[b] ^ v[c], 43); \
	v[a] +=  v[b] + (y ^ c2); \
	v[d] = ROTR64(v[d] ^ v[a], 5); \
	v[c] +=  v[d]; \
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


void vblake512_compress(unsigned long *h, unsigned long *mc)
{
	unsigned long v[16];
	unsigned long m[16] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    //#pragma unroll 8
	for (int i = 0; i < 8; i++)
	m[i] = mc[i];
	//#pragma unroll 8
	for (int i = 0; i < 8; i++) {
		v[i] = h[i];
		v[i + 8] = vBlake_iv[i];
	}
	v[12] ^= 64;
	v[14] ^= (ulong)(0xfffffffffffffffful);// (long)(-1);


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

}

unsigned long vBlake2(__global ulong *hi, ulong h7)
{
	unsigned long b[8];
	unsigned long h[8];
  // #pragma unroll 8
	for (int i = 0; i < 8; i++) {
		h[i] = vBlake_iv[i];
		b[i] = hi[i];
	}
	h[0] ^= (ulong)(0x01010000 ^ 0x18);

	b[7] = h7;

	vblake512_compress(h, b);

	return h[0];
}
//__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void kernel_vblake(__global uint *nonceStart, __global uint *nonceOut, __global unsigned long *hashStartOut, __global unsigned long  *headerIn)
{
	// Generate a unique starting nonce for each thread that doesn't overlap with the work of any other thread
	uint nonce = ((uint)get_global_id(0)&0xffffffffu) + nonceStart[0];

	unsigned long nonceHeaderSection = headerIn[7];

	
		nonceHeaderSection &= 0x00000000FFFFFFFFu;
		nonceHeaderSection |= (((unsigned long)nonce) << 32);

		unsigned long hashStart = vBlake2(headerIn, nonceHeaderSection);

		if ((hashStart & 0x00000000FFFFFFFFu) == 0) { // 2^32 difficulty
						
			
			if (hashStartOut[0] > hashStart || hashStartOut[0] == 0) {
				nonceOut[0] = nonce;
				hashStartOut[0] = hashStart;
			}

			
		}
	
}
