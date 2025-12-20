#ifndef __ATTENTION_FUNC_H__
#define __ATTENTION_FUNC_H__

#include "param.h"

// Attention implementation
void partial_attention(const m_axi_port_t *, const m_axi_port_t *, m_axi_port_t *);
void safe_softmax(m_axi_port_t *);
void final_attention(const m_axi_port_t *, const m_axi_port_t *, m_axi_port_t *);

// Attention kernel
void krnl_attention(const m_axi_port_t* input, m_axi_port_t* output);

#endif