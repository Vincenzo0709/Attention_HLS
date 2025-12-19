#ifndef __KRNL_UTIL__
#define __KRNL_UTIL__

#include "param.h"
#include <hls_math.h>       // for HLS optimized math functions

// Attention implementation
void partial_attention(const m_axi_port_t *, const m_axi_port_t *, m_axi_port_t *);
void safe_softmax(m_axi_port_t *);
void final_attention(m_axi_port_t *, const m_axi_port_t *, m_axi_port_t *);

// Attention kernel
void krnl_attention(const m_axi_port_t* input, m_axi_port_t* output);

#endif