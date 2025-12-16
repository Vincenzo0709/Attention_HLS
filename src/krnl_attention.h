#ifndef __KRNL_ATTENTION_H__
#define __KRNL_ATTENTION_H__

#include <hls_math.h>

void krnl_attention(const float*, float*);

#define B 1
#define T 32//1024
#define C 128//768
#define NH 12

#endif