#ifndef __KRNL_ATTENTION_H__
#define __KRNL_ATTENTION_H__

#include <hls_math.h>
#include <hls_vector.h>
#include <hls_half.h>

// +-------------------------------+
// | DIMENSION         | NOTATION  |
// |-------------------|-----------|
// | Batches           |     B     |
// | Tokens            |     T     |
// | Embeddings        |     C     |
// +-------------------------------+
#define B 1
#define T 1024
#define C 768

// INPUT TENSOR (BxTx3C)
// +-----------------------------------+
// |     Q     |     K     |     V     |
// |  (BxTxC)  |  (BxTxC)  |  (BxTxC)  |
// +-----------------------------------+
#define INPUT_SIZE      (B * T * 3 * C)

// OUTPUT TENSOR (BxTxC)
// +-----------+
// |     O     |
// |  (BxTxC)  |
// +-----------+
#define OUTPUT_SIZE     (B * T * C)

// Offsets to access (Q,K,V) from input
#define OFFSET_Q        0
#define OFFSET_K        (B*T*C)
#define OFFSET_V        (2*B*T*C)

// Interface is 512 bits
#define M_AXI_DWIDTH 512

#ifdef UINT8
    typedef uint8_t target_type_t;
#elif defined UINT16
    typedef uint16_t target_type_t;
#elif defined UINT32
    typedef uint32_t target_type_t;
#elif defined FLOAT16
    typedef hls::half target_type_t;
#elif defined FLOAT32
    typedef float target_type_t;
#elif defined DOUBLE
    typedef double target_type_t;
#else
    typedef float target_type_t;

#define INTERFACE_SIZE (M_AXI_DWIDTH / sizeof(target_type_t))

// Interface type
typedef hls::vector<target_type_t, INTERFACE_SIZE> m_axi_port_t;

// +------------------------------------------+
// | TENSOR          | NOTATION   | INDEXES   |
// |-----------------|------------|-----------|
// | Input           |  input     | [b][t][c] |
// | Output          |  output    | [b][t][c] |
// +------------------------------------------+


void krnl_attention(const m_axi_port_t*, m_axi_port_t*);

#endif