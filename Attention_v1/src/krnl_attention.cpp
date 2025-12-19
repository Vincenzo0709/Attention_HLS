#include "attention_func.h"

void krnl_attention(
                    const m_axi_port_t*     input,
                    m_axi_port_t*           output
                ) {

    // Interfaces specification
    #pragma HLS INTERFACE mode=m_axi port=input depth=INPUT_LINES bundle=gmem0 \
        max_read_burst_length=INTERFACE_SIZE \
        max_widen_bitwidth=512 \
        max_write_burst_length=INTERFACE_SIZE

    #pragma HLS INTERFACE mode=m_axi port=output depth=OUTPUT_LINES bundle=gmem0 \
        max_read_burst_length=INTERFACE_SIZE \
        max_widen_bitwidth=512 \
        max_write_burst_length=INTERFACE_SIZE

    // Zero-copy pointers
    const m_axi_port_t *Q_ptr = input + OFFSET_Q;
    const m_axi_port_t *K_ptr = input + OFFSET_K;
    const m_axi_port_t *V_ptr = input + OFFSET_V;

    // ------------------- //
    // Attention algorithm //
    // ------------------- //

    // Partial Attention result
    m_axi_port_t P[B*T*T / INTERFACE_SIZE];
    partial_attention(Q_ptr, K_ptr, P);

    // Safe Softmax
    safe_softmax(P);

    // Partial Attention * V
    final_attention(P, V_ptr, output);
    
}