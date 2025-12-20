#ifndef __PARAM_H__
#define __PARAM_H__

#include <hls_math.h>       // for HLS optimized math functions
#include <hls_vector.h>     // for hls::vector
#include <hls_half.h>       // for half float precision type

// +---------------------------------------+
// | DIMENSION         | NOTATION  | INDEX |
// |-------------------|-----------|-------|
// | Batches           |     B     |   b   |
// | Tokens            |     T     |   t   |
// | Embeddings        |     C     |   c   |
// +---------------------------------------+
#define B 1
#define T 1024 / 32
#define C (768 - 256) / 8

// Input tensor 3x(BxTxC)
#define INPUT_SIZE      3*(B*T*C)

// Output tensor (BxTxC)
#define OUTPUT_SIZE     (B*T*C)

// Interface is 512 bits
#define M_AXI_DWIDTH 512

// Different types are supported
#ifdef FLOAT16
    typedef hls::half target_type_t;
#elif defined FLOAT32
    typedef float target_type_t;
#elif defined DOUBLE
    typedef double target_type_t;
#else
    typedef float target_type_t;
#endif

// Interface size depends on target_type_t, so do number of lines in input and output
#define INTERFACE_SIZE          (M_AXI_DWIDTH / (sizeof(target_type_t) * 8))
#define INPUT_LINES             (INPUT_SIZE / INTERFACE_SIZE)
#define OUTPUT_LINES            (OUTPUT_SIZE / INTERFACE_SIZE)

// Offsets to access (Q,K,V) from input
#define OFFSET_Q        0
#define OFFSET_K        (B*T*C) / INTERFACE_SIZE
#define OFFSET_V        (2*B*T*C) / INTERFACE_SIZE

// Interface port type
typedef hls::vector<target_type_t, INTERFACE_SIZE> m_axi_port_t;

#endif