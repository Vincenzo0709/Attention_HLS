#include "attention_func.h"

void partial_attention(
                        const m_axi_port_t *Q,
                        const m_axi_port_t *K,
                        m_axi_port_t *P
                    ) {
    
    // Scaling factor
    target_type_t scale = 1.0 / hls::sqrt(C);

    // Local Q rows buffer
    m_axi_port_t Q_row[C / INTERFACE_SIZE];
    #pragma HLS array_partition variable=Q_row type=complete

    // Scanning batches
    for(int b=0; b<B; b++) {

        // Scanning tokens
        for(int t=0; t<T; t++) {

            // Q pre-fetch
            for(int i=0; i<C/INTERFACE_SIZE; i++) {
                #pragma HLS pipeline II=1

                int q_idx = ((b*T*C + t*C) / INTERFACE_SIZE) + i;
                Q_row[i] = Q[q_idx];
            }

            // Sums line is needed to store partial results in parallel
            m_axi_port_t sums;
            #pragma HLS array_partition variable=sums type=complete
            for(int i=0; i<INTERFACE_SIZE; i++) {
                #pragma HLS unroll
                sums[i] = 0.0f;
            }

            int sums_idx = 0;

            // Scanning only previous tokens for causality
            for(int t2=0; t2<=t; t2++) {
                #pragma HLS pipeline II=1

                target_type_t sum = 0.0f;

                // Scanning line by line, in order to force parallel reads for all elements on the line
                for (int line=0; line<C/INTERFACE_SIZE; line++) {
                    #pragma HLS unroll

                    // Buffering Q line
                    m_axi_port_t q_buff = Q_row[line];

                    // Buffering K line
                    #define K_IDX ((b*T*C + t2*C) / INTERFACE_SIZE) + line
                    m_axi_port_t k_buff;
                    k_buff = K[K_IDX];
                    
                    // Scanning each element on the line
                    for(int c=0; c<INTERFACE_SIZE; c++) {
                        #pragma HLS unroll

                        sum += q_buff[c] * k_buff[c];

                    }

                }

                // Storing sum into sums line after scaling
                sums[sums_idx++] = sum*scale;

                // Checking when at the end of a line for sums line. 
                //  In fact, t, the number of sums to do, is not necessarily multiple of INTERFACE_SIZE,
                //  and T is probably greater than INTERFACE_SIZE.
                if (sums_idx == INTERFACE_SIZE || t2 == t) {

                    int p_idx = (b*T*T + t*T + (t2 - sums_idx + 1)) / INTERFACE_SIZE;
                    P[p_idx] = sums;
                    sums_idx = 0;

                }

            }

        }

    }

}

void safe_softmax(m_axi_port_t *P) {

    // Local P rows buffer
    m_axi_port_t P_row[T/INTERFACE_SIZE];
    #pragma HLS array_partition variable=P_row type=complete

    // Scanning batches
    for(int b=0; b<B; b++) {

        // Scanning tokens
        for(int t=0; t<T; t++) {

            target_type_t max = -1e10;

            for (int i=0; i<T/INTERFACE_SIZE; i++) {
                #pragma HLS pipeline II=1

                int p_idx = ((b*T*T + t*T) / INTERFACE_SIZE) + i;
                P_row[i] = P[p_idx];
            }

            // Finding max value for safety
            // Scanning line by line, in order to force parallel reads for all elements on the line
            for (int line=0; line<T/INTERFACE_SIZE; line++) {
                #pragma HLS unroll

                m_axi_port_t p_buff = P_row[line];

                // Scanning line elements
                for (int t2=0; t2<INTERFACE_SIZE; t2++) {
                    #pragma HLS unroll

                    #define ELEM_IDX line*INTERFACE_SIZE + t2

                    // For causality the index must be <= t
                    if (p_buff[t2] > max && ELEM_IDX <= t) max = p_buff[t2];

                }

            }

            // Exponential sum after subtracting the max
            target_type_t expsum=0;

            // Scanning line by line, in order to force parallel reads for all elements on the line
            for (int line=0; line<T/INTERFACE_SIZE; line++) {
                #pragma HLS unroll

                m_axi_port_t p_buff = P_row[line];
                m_axi_port_t exp_buff;

                // Scanning line elements
                for (int t2=0; t2<INTERFACE_SIZE; t2++) {
                    #pragma HLS unroll
                    
                    // For causality the index must be <= t
                    if (ELEM_IDX <= t) {

                        target_type_t eval = hls::exp(p_buff[t2] - max);
                        exp_buff[t2] = eval;
                        expsum += eval;

                    } else {

                        exp_buff[t2] = 0.0f;

                    }

                }

                // Updating row buffer
                P_row[line] = exp_buff;

            
            }

            // Normalization
            target_type_t inv_expsum = 1.0 / expsum;
            // Scanning line by line, in order to force parallel reads for all elements on the line
            for (int line=0; line<T/INTERFACE_SIZE; line++) {
                #pragma HLS pipeline II=1

                m_axi_port_t p_buff = P_row[line];

                // Scanning line elements
                for (int t2=0; t2<INTERFACE_SIZE; t2++) {
                    #pragma HLS unroll

                    p_buff[t2] *= inv_expsum;

                }

                // Writing on local memory
                #define P_IDX ((b*T*T + t*T) / INTERFACE_SIZE) + line
                P[P_IDX] = p_buff;

            
            }

        }

    }

}

void final_attention(
                        const m_axi_port_t *P,
                        const m_axi_port_t *V,
                        m_axi_port_t *O
                    ) {

    // Local output rows buffer
    m_axi_port_t O_row[C/INTERFACE_SIZE];
    #pragma HLS array_partition variable=O_row type=complete
    
    // Scanning batches
    for(int b=0; b<B; b++) {

        // Scanning tokens
        for(int t=0; t<T; t++) {

            // Initializing to 0 local buffer
            for (int i=0; i<C/INTERFACE_SIZE; i++) {
                #pragma HLS unroll

                m_axi_port_t o_buff;
                for(int k=0; k<INTERFACE_SIZE; k++) o_buff[k] = 0.0f;
                O_row[i] = o_buff;

            }

            // Scanning line elements
            for(int t2=0; t2<=t; t2++) {
                #pragma HLS pipeline II=1
                
                #define P_LINE_IDX (b*T*T + t*T + t2) / INTERFACE_SIZE
                #define P_ELEM_IDX (b*T*T + t*T + t2) % INTERFACE_SIZE

                m_axi_port_t p_buff = P[P_LINE_IDX];
                target_type_t p_elem = p_buff[P_ELEM_IDX];

                // Scanning line by line, in order to force parallel reads for all elements on the line
                for (int line=0; line<C/INTERFACE_SIZE; line++) {
                    #pragma HLS unroll

                    m_axi_port_t sum_acc = O_row[line];

                    // Buffering V line
                    #define V_IDX ((b*T*C + t2*C) / INTERFACE_SIZE) + line
                    m_axi_port_t v_buff = V[V_IDX];

                    m_axi_port_t sum;

                    // Multiplying the element P[P_LINE_IDX][P_ELEM_IDX] by the line V[V_IDX]
                    for (int c=0; c<INTERFACE_SIZE; c++) {
                        #pragma HLS unroll
                        
                        sum[c] = sum_acc[c] + (p_elem * v_buff[c]);
                    
                    }

                    // Updating local buffer
                    O_row[line] = sum;

                }

            }

            // Storing the result
            for (int line=0; line<C/INTERFACE_SIZE; line++) {
                #pragma HLS pipeline II=1

                #define O_IDX ((b*T*C + t*C) / INTERFACE_SIZE) + line
                O[O_IDX] = O_row[line];

            }

        }

    }

}

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

    // Local URAM for P
    m_axi_port_t P[B*T*T / INTERFACE_SIZE];
    #pragma HLS BIND_STORAGE variable=P type=ram_2p impl=bram
    
    // Partial Attention result
    partial_attention(Q_ptr, K_ptr, P);

    // Safe Softmax
    safe_softmax(P);

    // Partial Attention * V
    final_attention(P, V_ptr, output);
    
}