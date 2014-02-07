/*
 * Copyright 2013 gerko.deroo@kangaderoo.nl
 * All rights reserved. 
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */


#include <immintrin.h>
#include "cpuminer-config.h"
#include "miner.h"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>



static inline void xor_salsa_sidm(__m128i *calc_18, __m128i *calc_13, __m128i *calc_9, __m128i *calc_7,
 								  const __m128i *calc_1, const __m128i *calc_4, const __m128i *calc_3, const __m128i *calc_2)
{
	int i;
	__m128i _calc;
	__m128i _shift_left;
	__m128i row1; // = _mm_xor_si128(*calc_18, *calc_1);;
	__m128i row2; // = _mm_xor_si128(*calc_7, *calc_2);;
	__m128i row3; // = _mm_xor_si128(*calc_9, *calc_3);;
	__m128i row4; // = _mm_xor_si128(*calc_13, *calc_4);;

	*calc_18 = _mm_xor_si128(*calc_18, *calc_1);
	*calc_7 = _mm_xor_si128(*calc_7, *calc_2);
	*calc_9 = _mm_xor_si128(*calc_9, *calc_3);
	*calc_13 = _mm_xor_si128(*calc_13, *calc_4);

	row1 = *calc_18;  //X[0]
	row2 = *calc_7;   //X[3]
	row3 = *calc_9;   //X[2]
	row4 = *calc_13;  //X[1]

	for (i = 0; i < 8; i += 2) {
		/* first row  X[3]=f(X0,X1) */
 		_calc = _mm_add_epi32(row1, row4);
		_shift_left = _mm_slli_epi32(_calc, 7);
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		row2 = _mm_xor_si128(row2, _calc);
		row2 = _mm_xor_si128(row2, _shift_left);

		/* second row X[2]=f(X3,X0) */
		_calc = _mm_add_epi32(row2, row1);
		_shift_left = _mm_slli_epi32(_calc, 9);
		_calc = _mm_srli_epi32(_calc,(32 - 9));
		row3 = _mm_xor_si128(row3, _calc);
		row3 = _mm_xor_si128(row3, _shift_left);

		/* third row X[1]=f(X2,X3) */
		_calc = _mm_add_epi32(row3, row2);
		_shift_left = _mm_slli_epi32(_calc, 13);
		_calc = _mm_srli_epi32(_calc,(32 - 13));
		row4 = _mm_xor_si128(row4, _calc);
		row4 = _mm_xor_si128(row4, _shift_left);

		/* fourth row X[0]=f(X1,X2) */
		_calc = _mm_add_epi32(row4, row3);
		_shift_left = _mm_slli_epi32(_calc, 18);
		_calc = _mm_srli_epi32(_calc,(32 - 18));
		row1 = _mm_xor_si128(row1, _calc);
		row1 = _mm_xor_si128(row1, _shift_left);

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2 = _mm_shuffle_epi32(row2,0x93);
		row3 = _mm_shuffle_epi32(row3,0x4e);
		row4 = _mm_shuffle_epi32(row4,0x39);
	// end transpose

		// switch *calc_13 and * calc_7 usage compared to rows
		/* first column X[1]=f(X0,X3) */
		_calc = _mm_add_epi32(row1, row2);
		_shift_left = _mm_slli_epi32(_calc, 7);
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		row4 = _mm_xor_si128(row4, _calc);
		row4 = _mm_xor_si128(row4, _shift_left);

		/* second column X[2]=f(X1,X0) */
		_calc = _mm_add_epi32(row4, row1);
		_shift_left = _mm_slli_epi32(_calc, 9);
		_calc = _mm_srli_epi32(_calc,(32 - 9));
		row3 = _mm_xor_si128(row3, _calc);
		row3 = _mm_xor_si128(row3, _shift_left);

		/* third column  X[3]=f(X2,X1) */
		_calc = _mm_add_epi32(row3, row4);
		_shift_left = _mm_slli_epi32(_calc, 13);
		_calc = _mm_srli_epi32(_calc,(32 - 13));
		row2 = _mm_xor_si128(row2, _calc);
		row2 = _mm_xor_si128(row2, _shift_left);

		/* fourth column  X[0]=f(X3,X2) */
		_calc = _mm_add_epi32(row2, row3);
		_shift_left = _mm_slli_epi32(_calc, 18);
		_calc = _mm_srli_epi32(_calc,(32 - 18));
		row1 = _mm_xor_si128(row1, _calc);
		row1 = _mm_xor_si128(row1, _shift_left);

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2 = _mm_shuffle_epi32(row2,0x39);
		row3 = _mm_shuffle_epi32(row3,0x4e);
		row4 = _mm_shuffle_epi32(row4,0x93);
	// end transpose
	}
	*calc_18 = _mm_add_epi32(*calc_18,row1);
	*calc_7 = _mm_add_epi32(*calc_7, row2);
	*calc_9 = _mm_add_epi32(*calc_9, row3);
	*calc_13 = _mm_add_epi32(*calc_13, row4);
}

static inline void scrypt_core_sidm(uint32_t *X /*, uint32_t *V*/)
{
	uint32_t i, j;

	__m128i scratch[2048 * 8];
	__m128i *SourcePtr = (__m128i*) X;
	uint32_t row1[4] __attribute__((aligned(16)));
	uint32_t row2[4] __attribute__((aligned(16)));
	uint32_t row3[4] __attribute__((aligned(16)));
	uint32_t row4[4] __attribute__((aligned(16)));

	uint32_t row11[4] __attribute__((aligned(16)));
	uint32_t row21[4] __attribute__((aligned(16)));
	uint32_t row31[4] __attribute__((aligned(16)));
	uint32_t row41[4] __attribute__((aligned(16)));

	__m128i *calc_1 = (__m128i*) row1;
	__m128i *calc_2 = (__m128i*) row2;
	__m128i *calc_3 = (__m128i*) row3;
	__m128i *calc_4 = (__m128i*) row4;

	__m128i *calc_11 = (__m128i*) row11;
	__m128i *calc_21 = (__m128i*) row21;
	__m128i *calc_31 = (__m128i*) row31;
	__m128i *calc_41 = (__m128i*) row41;

	__m128i _calc5;
	__m128i _calc6;
	__m128i _calc7;
	__m128i _calc8;

	// working with multiple pointers for the scratch-pad results in minimized instruction count.
    __m128i *scratchPrt1 = &scratch[0];
    __m128i *scratchPrt2 = &scratch[1];
    __m128i *scratchPrt3 = &scratch[2];
    __m128i *scratchPrt4 = &scratch[3];
    __m128i *scratchPrt11 = &scratch[4];
    __m128i *scratchPrt21 = &scratch[5];
    __m128i *scratchPrt31 = &scratch[6];
    __m128i *scratchPrt41 = &scratch[7];

	/* transpose the data from *X */
	_calc5 =_mm_blend_epi16(SourcePtr[4], SourcePtr[6], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[5], SourcePtr[7], 0x0f);
	_calc7 =_mm_blend_epi16(SourcePtr[6], SourcePtr[4], 0xf0);
	_calc8 =_mm_blend_epi16(SourcePtr[7], SourcePtr[5], 0x0f);
	*calc_11 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_21 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_31 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_41 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(SourcePtr[0], SourcePtr[2], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[1], SourcePtr[3], 0x0f);
	_calc7 =_mm_blend_epi16(SourcePtr[2], SourcePtr[0], 0xf0);
	_calc8 =_mm_blend_epi16(SourcePtr[3], SourcePtr[1], 0x0f);
	*calc_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	for (i = 0; i < 2048; i++) {
		scratch[i * 8 + 0] = *calc_1;
		scratch[i * 8 + 1] = *calc_2;
		scratch[i * 8 + 2] = *calc_3;
		scratch[i * 8 + 3] = *calc_4;
		scratch[i * 8 + 4] = *calc_11;
		scratch[i * 8 + 5] = *calc_21;
		scratch[i * 8 + 6] = *calc_31;
		scratch[i * 8 + 7] = *calc_41;

		xor_salsa_sidm(calc_1, calc_2, calc_3, calc_4,calc_11,calc_21,calc_31,calc_41);
		xor_salsa_sidm(calc_11,calc_21,calc_31,calc_41,calc_1, calc_2, calc_3, calc_4);
	}
	for (i = 0; i < 2048; i++) {
		j = 8 * (_mm_extract_epi16(*calc_11,0x00) & 2047);

		*calc_1 ^=  scratchPrt1[j];
		*calc_2 ^=  scratchPrt2[j];
		*calc_3 ^=  scratchPrt3[j];
		*calc_4 ^=  scratchPrt4[j];
		*calc_11 ^= scratchPrt11[j];
		*calc_21 ^=  scratchPrt21[j];
		*calc_31 ^=  scratchPrt31[j];
		*calc_41 ^=  scratchPrt41[j];

		xor_salsa_sidm(calc_1, calc_2, calc_3, calc_4,calc_11,calc_21,calc_31,calc_41);
		xor_salsa_sidm(calc_11,calc_21,calc_31,calc_41,calc_1, calc_2, calc_3, calc_4);
	}

	_calc5 =_mm_blend_epi16(*calc_11, *calc_31, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_21, *calc_41, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_31, *calc_11, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_41, *calc_21, 0x0f);
	SourcePtr[4] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	SourcePtr[5] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	SourcePtr[6] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	SourcePtr[7] = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1, *calc_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2, *calc_4, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3, *calc_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4, *calc_2, 0x0f);
	SourcePtr[0] = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	SourcePtr[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	SourcePtr[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	SourcePtr[3] = _mm_blend_epi16(_calc8, _calc7, 0xcc);
}

static inline void xor_salsa_sidm_3way(__m128i *calc_11, __m128i *calc_21, __m128i *calc_31)
{
	int i;
	__m128i _calc_x1;
	__m128i _calc_x2;
	__m128i _calc_x3;
	__m128i _shift_left;
	__m128i X1[4];
	__m128i X2[4];
	__m128i X3[4];

	X1[0] = calc_11[0];
	X1[1] = calc_11[1];
	X1[2] = calc_11[2];
	X1[3] = calc_11[3];

	X2[0] = calc_21[0];
	X2[1] = calc_21[1];
	X2[2] = calc_21[2];
	X2[3] = calc_21[3];

	X3[0] = calc_31[0];
	X3[1] = calc_31[1];
	X3[2] = calc_31[2];
	X3[3] = calc_31[3];

	for (i = 0; i < 8; i += 2) {
		/* first row  X[3]=f(X0,X1) */
 		_calc_x1 = _mm_add_epi32(X1[0], X1[1]);     //X[0] and X[1]
 		_calc_x2 = _mm_add_epi32(X2[0], X2[1]);     //X[0] and X[1]
 		_calc_x3 = _mm_add_epi32(X3[0], X3[1]);     //X[0] and X[1]
		_shift_left = _mm_slli_epi32(_calc_x1, 7);
		X1[3] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x2, 7);
		X2[3] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x3, 7);
		X3[3] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 7));
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 7));
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 7));
		X1[3] ^= _calc_x1;
		X2[3] ^= _calc_x2;
		X3[3] ^= _calc_x3;

		/* second rows X[2]=f(X3,X0) */
 		_calc_x1 = _mm_add_epi32(X1[3], X1[0]);     //X[3] and X[0]
 		_calc_x2 = _mm_add_epi32(X2[3], X2[0]);     //X[3] and X[0]
 		_calc_x3 = _mm_add_epi32(X3[3], X3[0]);     //X[3] and X[0]
		_shift_left = _mm_slli_epi32(_calc_x1, 9);
		X1[2] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x2, 9);
		X2[2] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x3, 9);
		X3[2] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 9));
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 9));
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 9));
		X1[2] ^= _calc_x1;
		X2[2] ^= _calc_x2;
		X3[2] ^= _calc_x3;

		/* third rows X[1]=f(X2,X3) */
 		_calc_x1 = _mm_add_epi32(X1[2], X1[3]);     //X[2] and X[3]
 		_calc_x2 = _mm_add_epi32(X2[2], X2[3]);     //X[2] and X[3]
 		_calc_x3 = _mm_add_epi32(X3[2], X3[3]);     //X[2] and X[3]
		_shift_left = _mm_slli_epi32(_calc_x1, 13);
		X1[1] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x2, 13);
		X2[1] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x3, 13);
		X3[1] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 13));
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 13));
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 13));
		X1[1] ^= _calc_x1;
		X2[1] ^= _calc_x2;
		X3[1] ^= _calc_x3;

		/* fourth rows X[0]=f(X1,X2) */
 		_calc_x1 = _mm_add_epi32(X1[1], X1[2]);     //X[1] and X[2]
 		_calc_x2 = _mm_add_epi32(X2[1], X2[2]);     //X[1] and X[2]
 		_calc_x3 = _mm_add_epi32(X3[1], X3[2]);     //X[1] and X[2]
		_shift_left = _mm_slli_epi32(_calc_x1, 18);
		X1[0] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x2, 18);
		X2[0] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x3, 18);
		X3[0] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 18));
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 18));
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 18));
		X1[0] ^= _calc_x1;
		X2[0] ^= _calc_x2;
		X3[0] ^= _calc_x3;

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		X1[3] = _mm_shuffle_epi32(X1[3],0x93);    //x[3]
		X2[3] = _mm_shuffle_epi32(X2[3],0x93);    //x[3]
		X3[3] = _mm_shuffle_epi32(X3[3],0x93);    //x[3]
		X1[2] = _mm_shuffle_epi32(X1[2],0x4e);    //x[2]
		X2[2] = _mm_shuffle_epi32(X2[2],0x4e);    //x[2]
		X3[2] = _mm_shuffle_epi32(X3[2],0x4e);    //x[2]
		X1[1] = _mm_shuffle_epi32(X1[1],0x39);    //x[1]
		X2[1] = _mm_shuffle_epi32(X2[1],0x39);    //x[1]
		X3[1] = _mm_shuffle_epi32(X3[1],0x39);    //x[1]
	// end transpose

		// switch *calc_13 and * calc_7 usage compared to rows
		/* first column X[1]=f(X0,X3) */
 		_calc_x1 = _mm_add_epi32(X1[0], X1[3]);     //X[0] and X[3]
 		_calc_x2 = _mm_add_epi32(X2[0], X2[3]);     //X[0] and X[3]
 		_calc_x3 = _mm_add_epi32(X3[0], X3[3]);     //X[0] and X[3]
		_shift_left = _mm_slli_epi32(_calc_x1, 7);
		X1[1] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x2, 7);
		X2[1] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x3, 7);
		X3[1] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 7));
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 7));
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 7));
		X1[1] ^= _calc_x1;
		X2[1] ^= _calc_x2;
		X3[1] ^= _calc_x3;

		/* second column X[2]=f(X1,X0) */
 		_calc_x1 = _mm_add_epi32(X1[1], X1[0]);     //X[1] and X[0]
 		_calc_x2 = _mm_add_epi32(X2[1], X2[0]);     //X[1] and X[0]
 		_calc_x3 = _mm_add_epi32(X3[1], X3[0]);     //X[1] and X[0]
		_shift_left = _mm_slli_epi32(_calc_x1, 9);
		X1[2] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x2, 9);
		X2[2] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x3, 9);
		X3[2] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 9));
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 9));
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 9));
		X1[2] ^= _calc_x1;
		X2[2] ^= _calc_x2;
		X3[2] ^= _calc_x3;

		/* third column  X[3]=f(X2,X1) */
 		_calc_x1 = _mm_add_epi32(X1[2], X1[1]);     //X[2] and X[1]
 		_calc_x2 = _mm_add_epi32(X2[2], X2[1]);     //X[2] and X[1]
 		_calc_x3 = _mm_add_epi32(X3[2], X3[1]);     //X[2] and X[1]
		_shift_left = _mm_slli_epi32(_calc_x1, 13);
		X1[3] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x2, 13);
		X2[3] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x3, 13);
		X3[3] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 13));
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 13));
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 13));
		X1[3] ^= _calc_x1;
		X2[3] ^= _calc_x2;
		X3[3] ^= _calc_x3;

		/* fourth column  X[0]=f(X3,X2) */
 		_calc_x1 = _mm_add_epi32(X1[3], X1[2]);     //X[3] and X[2]
 		_calc_x2 = _mm_add_epi32(X2[3], X2[2]);     //X[3] and X[2]
 		_calc_x3 = _mm_add_epi32(X3[3], X3[2]);     //X[3] and X[2]
		_shift_left = _mm_slli_epi32(_calc_x1, 18);
		X1[0] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x2, 18);
		X2[0] ^= _shift_left;
		_shift_left = _mm_slli_epi32(_calc_x3, 18);
		X3[0] ^= _shift_left;
		_calc_x1 = _mm_srli_epi32(_calc_x1,(32 - 18));
		_calc_x2 = _mm_srli_epi32(_calc_x2,(32 - 18));
		_calc_x3 = _mm_srli_epi32(_calc_x3,(32 - 18));
		X1[0] ^= _calc_x1;		//X[0]
		X2[0] ^= _calc_x2;		//X[0]
		X3[0] ^= _calc_x3;		//X[0]

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		X1[3] = _mm_shuffle_epi32(X1[3],0x39);    //x[3]
		X2[3] = _mm_shuffle_epi32(X2[3],0x39);    //x[3]
		X3[3] = _mm_shuffle_epi32(X3[3],0x39);    //x[3]
		X1[2] = _mm_shuffle_epi32(X1[2],0x4e);    //x[2]
		X2[2] = _mm_shuffle_epi32(X2[2],0x4e);    //x[2]
		X3[2] = _mm_shuffle_epi32(X3[2],0x4e);    //x[2]
		X1[1] = _mm_shuffle_epi32(X1[1],0x93);    //x[1]
		X2[1] = _mm_shuffle_epi32(X2[1],0x93);    //x[1]
		X3[1] = _mm_shuffle_epi32(X3[1],0x93);    //x[1]

	// end transpose
	}

	calc_11[0] = _mm_add_epi32(calc_11[0], X1[0]);
	calc_11[1] = _mm_add_epi32(calc_11[1], X1[1]);
	calc_11[2] = _mm_add_epi32(calc_11[2], X1[2]);
	calc_11[3] = _mm_add_epi32(calc_11[3], X1[3]);

	calc_21[0] = _mm_add_epi32(calc_21[0], X2[0]);
	calc_21[1] = _mm_add_epi32(calc_21[1], X2[1]);
	calc_21[2] = _mm_add_epi32(calc_21[2], X2[2]);
	calc_21[3] = _mm_add_epi32(calc_21[3], X2[3]);

	calc_31[0] = _mm_add_epi32(calc_31[0], X3[0]);
	calc_31[1] = _mm_add_epi32(calc_31[1], X3[1]);
	calc_31[2] = _mm_add_epi32(calc_31[2], X3[2]);
	calc_31[3] = _mm_add_epi32(calc_31[3], X3[3]);

}


static inline void scrypt_core_sidm_3way(uint32_t *X, uint32_t NFactor /*, uint32_t *V*/)
{
	uint32_t i, j;
	uint32_t N;

	__m128i *SourcePtr = (__m128i*) X;
	__m128i X11[4];
	__m128i X12[4];
	__m128i X21[4];
	__m128i X22[4];
	__m128i X31[4];
	__m128i X32[4];
	__m128i *scratch;

	__m128i *calc_11 = (__m128i*) X11;
	__m128i *calc_21 = (__m128i*) X21;
	__m128i *calc_31 = (__m128i*) X31;
	__m128i *calc_12 = (__m128i*) X12;
	__m128i *calc_22 = (__m128i*) X22;
	__m128i *calc_32 = (__m128i*) X32;

	__m128i _calc5;
	__m128i _calc6;
	__m128i _calc7;
//	__m128i _calc8;

	N = 1 << ( NFactor + 1);
	scratch = malloc(N * 3 * 8 * sizeof(__m128i));
    N = N - 1;

	// working with multiple pointers for the scratch-pad results in minimized instruction count.
//    __m128i *scratchPrt1 = &scratch[0];
//    __m128i *scratchPrt2 = &scratch[1];
//    __m128i *scratchPrt3 = &scratch[2];
//    __m128i *scratchPrt4 = &scratch[3];
//    __m128i *scratchPrt5 = &scratch[4];
//    __m128i *scratchPrt6 = &scratch[5];
//    __m128i *scratchPrt7 = &scratch[6];
//    __m128i *scratchPrt8 = &scratch[7];

    __m128i *scratchPrt1 = scratch;
    __m128i *scratchPrt2 = scratch+1;
    __m128i *scratchPrt3 = scratch+2;
    __m128i *scratchPrt4 = scratch+3;
    __m128i *scratchPrt5 = scratch+4;
    __m128i *scratchPrt6 = scratch+5;
    __m128i *scratchPrt7 = scratch+6;
    __m128i *scratchPrt8 = scratch+7;

	/* transpose the data from *X1x */
	_calc5 =_mm_blend_epi16(SourcePtr[0], SourcePtr[2], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[1], SourcePtr[3], 0x0f);
	calc_11[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(SourcePtr[2], SourcePtr[0], 0xf0);
	calc_11[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(SourcePtr[3], SourcePtr[1], 0x0f);
	calc_11[0] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	calc_11[3] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(SourcePtr[4], SourcePtr[6], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[5], SourcePtr[7], 0x0f);
	calc_12[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(SourcePtr[6], SourcePtr[4], 0xf0);
	calc_12[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(SourcePtr[7], SourcePtr[5], 0x0f);
	calc_12[0] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	calc_12[3] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	/* transpose the data from *X2x */
	_calc5 =_mm_blend_epi16(SourcePtr[8], SourcePtr[10], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[9], SourcePtr[11], 0x0f);
	calc_21[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(SourcePtr[10], SourcePtr[8], 0xf0);
	calc_21[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(SourcePtr[11], SourcePtr[9], 0x0f);
	calc_21[0] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	calc_21[3] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(SourcePtr[12], SourcePtr[14], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[13], SourcePtr[15], 0x0f);
	calc_22[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(SourcePtr[14], SourcePtr[12], 0xf0);
	calc_22[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(SourcePtr[15], SourcePtr[13], 0x0f);
	calc_22[0] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	calc_22[3] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	/* transpose the data from *X3x */
	_calc5 =_mm_blend_epi16(SourcePtr[16], SourcePtr[18], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[17], SourcePtr[19], 0x0f);
	calc_31[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(SourcePtr[18], SourcePtr[16], 0xf0);
	calc_31[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(SourcePtr[19], SourcePtr[17], 0x0f);
	calc_31[0] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	calc_31[3] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(SourcePtr[20], SourcePtr[22], 0xf0);
	_calc6 =_mm_blend_epi16(SourcePtr[21], SourcePtr[23], 0x0f);
	calc_32[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(SourcePtr[22], SourcePtr[20], 0xf0);
	calc_32[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(SourcePtr[23], SourcePtr[21], 0x0f);
	calc_32[0] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	calc_32[3] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	for (i = 0; i <= N; i++) {
		for (j=0; j<4; j++){
			scratch[i * 24 +  0 + j] = calc_11[j];
			scratch[i * 24 +  4 + j] = calc_12[j];
			scratch[i * 24 +  8 + j] = calc_21[j];
			scratch[i * 24 + 12 + j] = calc_22[j];
			scratch[i * 24 + 16 + j] = calc_31[j];
			scratch[i * 24 + 20 + j] = calc_32[j];
		}
		calc_11[0] ^= calc_12[0];
		calc_11[1] ^= calc_12[1];
		calc_11[2] ^= calc_12[2];
		calc_11[3] ^= calc_12[3];

		calc_21[0] ^= calc_22[0];
		calc_21[1] ^= calc_22[1];
		calc_21[2] ^= calc_22[2];
		calc_21[3] ^= calc_22[3];

		calc_31[0] ^= calc_32[0];
		calc_31[1] ^= calc_32[1];
		calc_31[2] ^= calc_32[2];
		calc_31[3] ^= calc_32[3];

		xor_salsa_sidm_3way(calc_11, calc_21, calc_31);

		calc_12[0] ^= calc_11[0];
		calc_12[1] ^= calc_11[1];
		calc_12[2] ^= calc_11[2];
		calc_12[3] ^= calc_11[3];

		calc_22[0] ^= calc_21[0];
		calc_22[1] ^= calc_21[1];
		calc_22[2] ^= calc_21[2];
		calc_22[3] ^= calc_21[3];

		calc_32[0] ^= calc_31[0];
		calc_32[1] ^= calc_31[1];
		calc_32[2] ^= calc_31[2];
		calc_32[3] ^= calc_31[3];

		xor_salsa_sidm_3way(calc_12, calc_22, calc_32);
	}
	for (i = 0; i <= N; i++) {
		j = 24 * (_mm_extract_epi16(calc_12[0],0x00) & N);

		calc_11[0] ^=  scratchPrt1[j];
		calc_11[1] ^=  scratchPrt2[j];
		calc_11[2] ^=  scratchPrt3[j];
		calc_11[3] ^=  scratchPrt4[j];
		calc_12[0] ^=  scratchPrt5[j];
		calc_12[1] ^=  scratchPrt6[j];
		calc_12[2] ^=  scratchPrt7[j];
		calc_12[3] ^=  scratchPrt8[j];

		j = 8 + 24 * (_mm_extract_epi16(calc_22[0],0x00) & N);

		calc_21[0] ^=  scratchPrt1[j];
		calc_21[1] ^=  scratchPrt2[j];
		calc_21[2] ^=  scratchPrt3[j];
		calc_21[3] ^=  scratchPrt4[j];
		calc_22[0] ^=  scratchPrt5[j];
		calc_22[1] ^=  scratchPrt6[j];
		calc_22[2] ^=  scratchPrt7[j];
		calc_22[3] ^=  scratchPrt8[j];

		j = 16 + 24 * (_mm_extract_epi16(calc_32[0],0x00) & N);

		calc_31[0] ^=  scratchPrt1[j];
		calc_31[1] ^=  scratchPrt2[j];
		calc_31[2] ^=  scratchPrt3[j];
		calc_31[3] ^=  scratchPrt4[j];
		calc_32[0] ^=  scratchPrt5[j];
		calc_32[1] ^=  scratchPrt6[j];
		calc_32[2] ^=  scratchPrt7[j];
		calc_32[3] ^=  scratchPrt8[j];

		calc_11[0] ^= calc_12[0];
		calc_11[1] ^= calc_12[1];
		calc_11[2] ^= calc_12[2];
		calc_11[3] ^= calc_12[3];

		calc_21[0] ^= calc_22[0];
		calc_21[1] ^= calc_22[1];
		calc_21[2] ^= calc_22[2];
		calc_21[3] ^= calc_22[3];

		calc_31[0] ^= calc_32[0];
		calc_31[1] ^= calc_32[1];
		calc_31[2] ^= calc_32[2];
		calc_31[3] ^= calc_32[3];

		xor_salsa_sidm_3way(calc_11, calc_21, calc_31);

		calc_12[0] ^= calc_11[0];
		calc_12[1] ^= calc_11[1];
		calc_12[2] ^= calc_11[2];
		calc_12[3] ^= calc_11[3];

		calc_22[0] ^= calc_21[0];
		calc_22[1] ^= calc_21[1];
		calc_22[2] ^= calc_21[2];
		calc_22[3] ^= calc_21[3];

		calc_32[0] ^= calc_31[0];
		calc_32[1] ^= calc_31[1];
		calc_32[2] ^= calc_31[2];
		calc_32[3] ^= calc_31[3];

		xor_salsa_sidm_3way(calc_12, calc_22, calc_32);
	}
// return the valueś to X
	_calc5 =_mm_blend_epi16(calc_11[0], calc_11[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_11[1], calc_11[3], 0x0f);
	SourcePtr[1] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(calc_11[2], calc_11[0], 0xf0);
	SourcePtr[2] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(calc_11[3], calc_11[1], 0x0f);
	SourcePtr[0] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	SourcePtr[3] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_12[0], calc_12[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_12[1], calc_12[3], 0x0f);
	SourcePtr[5] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(calc_12[2], calc_12[0], 0xf0);
	SourcePtr[6] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(calc_12[3], calc_12[1], 0x0f);
	SourcePtr[4] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	SourcePtr[7] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_21[0], calc_21[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_21[1], calc_21[3], 0x0f);
	SourcePtr[9] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(calc_21[2], calc_21[0], 0xf0);
	SourcePtr[10] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(calc_21[3], calc_21[1], 0x0f);
	SourcePtr[8] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	SourcePtr[11] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_22[0], calc_22[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_22[1], calc_22[3], 0x0f);
	SourcePtr[13] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(calc_22[2], calc_22[0], 0xf0);
	SourcePtr[14] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(calc_22[3], calc_22[1], 0x0f);
	SourcePtr[12] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	SourcePtr[15] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_31[0], calc_31[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_31[1], calc_31[3], 0x0f);
	SourcePtr[17] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(calc_31[2], calc_31[0], 0xf0);
	SourcePtr[18] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(calc_31[3], calc_31[1], 0x0f);
	SourcePtr[16] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	SourcePtr[19] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(calc_32[0], calc_32[2], 0xf0);
	_calc6 =_mm_blend_epi16(calc_32[1], calc_32[3], 0x0f);
	SourcePtr[21] = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	_calc7 =_mm_blend_epi16(calc_32[2], calc_32[0], 0xf0);
	SourcePtr[22] = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	_calc6 =_mm_blend_epi16(calc_32[3], calc_32[1], 0x0f);
	SourcePtr[20] = _mm_blend_epi16(_calc5, _calc6, 0xcc);
	SourcePtr[23] = _mm_blend_epi16(_calc6, _calc7, 0xcc);

	free (scratch);
}

