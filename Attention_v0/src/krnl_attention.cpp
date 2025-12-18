#include "krnl_attention.h"

void load_input(
                    const m_axi_port_t *in,
                    target_type_t mat[B][T][C],
                    const int offset
                ) {
    
    // Starting index for m_axi_port_t array
    int curr_line = offset / INTERFACE_SIZE;

    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            for (int line=0; line<C/INTERFACE_SIZE; line++) {

                // Temporary buffer to access input line
                m_axi_port_t buff = in[curr_line++];

                for(int c=0; c<INTERFACE_SIZE; c++) {
                    
                    mat[b][t][c + line*INTERFACE_SIZE] = buff[c];
                
                }

            }

        }

    }

}

void store_output(
                    m_axi_port_t *out,
                    target_type_t mat[B][T][C]
                ) {

    // Starting index for m_axi_port_t array
    int curr_line = 0;

    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            for (int line=0; line<C/INTERFACE_SIZE; line++) {

                // Temporary buffer to store lines
                m_axi_port_t buff;
                
                for(int c=0; c<INTERFACE_SIZE; c++) {
                    
                    buff[c] = mat[b][t][c + line*INTERFACE_SIZE];
                
                }

                out[curr_line++] = buff;

            }

        }

    }

}

void partial_attention(
                        const target_type_t Q[B][T][C],
                        const target_type_t K[B][T][C],
                        target_type_t P[B][T][T]
                    ) {
    
    // Scaling factor
    target_type_t scale = 1.0 / hls::sqrt((target_type_t)C);

    // QK^T causal
    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            // Scanning only prevoius tokens for causality
            for(int t2=0; t2<=t; t2++) {

                target_type_t sum = 0.0f;

                for(int c=0; c<C; c++) {
                    
                    sum += Q[b][t][c] * K[b][t2][c];

                }

                // Scaling and storing partial result
                P[b][t][t2] = sum * scale;

            }

        }

    }
}

void safe_softmax(target_type_t P[B][T][T]) {

    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            target_type_t max = -1e10;

            // Max value for safety
            for(int t2=0; t2<=t; t2++) {

                if (P[b][t][t2] > max) max = P[b][t][t2];

            }

            // Exponential sum after subtracting the max
            target_type_t expsum=0;
            for(int t2=0; t2<=t; t2++) {

                target_type_t eval = hls::exp(P[b][t][t2] - max);
                P[b][t][t2] = eval;
                expsum += eval;

            }

            // Normalization
            target_type_t inv_expsum = 1.0 / expsum;
            for(int t2=0; t2<=t; t2++) {

                P[b][t][t2] *= inv_expsum;

            }

        }

    }

}

void final_attention(
                        const target_type_t P[B][T][T],
                        const target_type_t V[B][T][C],
                        target_type_t O[B][T][C]
                    ) {

    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            // Scanning each embedding
            for(int c=0; c<C; c++) {

                target_type_t sum = 0.0f;

                for(int t2=0; t2<=t; t2++) {
                    
                    sum += P[b][t][t2] * V[b][t2][c];

                }

                // Storing the result
                O[b][t][c] = sum;

            }

        }

    }

}

void attention(
                target_type_t Q[B][T][C],
                target_type_t K[B][T][C],
                target_type_t V[B][T][C],
                target_type_t O[B][T][C]
            ) {

    // Partial Attention result
    target_type_t P[B][T][T];
    partial_attention(Q, K, P);

    // Safe Softmax
    safe_softmax(P);

    // Attention * V
    final_attention(P, V, O);

}


void krnl_attention(
                    const m_axi_port_t*     input,
                    m_axi_port_t*           output
                ) {

    #pragma HLS INTERFACE mode=m_axi port=input depth=INPUT_LINES bundle=gmem0 \
        max_read_burst_length=INTERFACE_SIZE \
        max_widen_bitwidth=512 \
        max_write_burst_length=INTERFACE_SIZE

    #pragma HLS INTERFACE mode=m_axi port=output depth=OUTPUT_LINES bundle=gmem0 \
        max_read_burst_length=INTERFACE_SIZE \
        max_widen_bitwidth=512 \
        max_write_burst_length=INTERFACE_SIZE

    // Local buffers
    target_type_t Q[B][T][C];
    target_type_t K[B][T][C];
    target_type_t V[B][T][C];

    target_type_t O[B][T][C];

    // Load from interface to local buffers
    load_input(input, Q, OFFSET_Q);
    load_input(input, K, OFFSET_K);
    load_input(input, V, OFFSET_V);

    // Attention algorithm
    attention(Q, K, V, O);

    // Store from local buffer to interface
    store_output(output, O);
    
}