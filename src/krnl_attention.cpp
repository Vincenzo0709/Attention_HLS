#include "krnl_attention.h"

void load_input(const float16_t* in, float mat[B][T][C], const int offset) {

    int b=0, t=0, c=0;
    float16_t buff;
    
    for(int l=0; l<INPUT_MM_SIZE; l++) {

        buff=in[l + offset];

        if (c==C) {
            c=0;
            t++;

            if (t==T) {
                t=0;
                b++;
            }

        }
                
        for (int k=0; k<INTERFACE_SIZE; k++) {
            #pragma HLS UNROLL factor=INTERFACE_SIZE

            mat[b][t][c] = ((float *)(&buff))[k];
            c++;

        }
    }

}

void store_output(float *out, float O[B][T][C])
{

    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            for(int c=0; c<C; c++) {

                // The current index points to:
                // - b sequences ahead (sized T*C);
                // - t tokens ahead (sized C);
                // - c elements ahead.
                int idx=b*T*C + t*C + c;
                
                out[idx]= O[b][t][c];
            
            }

        }

    }

}

void partial_attention(const float Q[B][T][C], const float K[B][T][C], float P[B][T][T])
{
    
    // Scaling factor
    float scale = 1.0 / hls::sqrt(C);

    // QK^T causal
    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            // Scanning only prevoius tokens for causality
            for(int t2=0; t2<=t; t2++) {

                float sum = 0.0f;

                for(int c=0; c<C; c++) {
                    
                    sum += Q[b][t][c] * K[b][t2][c];

                }

                // Scaling and storing partial result
                P[b][t][t2] = sum * scale;

            }

        }

    }
}

void safe_softmax(float P[B][T][T])
{

    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            float max = -1e10;

            // Max value for safety
            for(int t2=0; t2<=t; t2++) {

                if (P[b][t][t2] > max) max = P[b][t][t2];

            }

            // Exponential sum after subtracting the max
            float expsum=0;
            for(int t2=0; t2<=t; t2++) {

                float eval = hls::exp(P[b][t][t2] - max);
                P[b][t][t2] = eval;
                expsum += eval;

            }

            // Normalization
            float inv_expsum = 1.0 / expsum;
            for(int t2=0; t2<=t; t2++) {

                P[b][t][t2] *= inv_expsum;

            }

        }

    }

}

void final_attention(const float P[B][T][T], const float V[B][T][C], float O[B][T][C])
{

    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            // Scanning each embedding
            for(int c=0; c<C; c++) {

                float sum = 0.0f;

                for(int t2=0; t2<=t; t2++) {
                    
                    sum += P[b][t][t2] * V[b][t2][c];

                }

                // Storing the result
                O[b][t][c] = sum;

            }

        }

    }

}

void attention(float Q[B][T][C], float K[B][T][C], float V[B][T][C], float O[B][T][C])
{

    // Partial Attention result
    float P[B][T][T];
    partial_attention(Q, K, P);

    // Safe Softmax
    safe_softmax(P);

    // Attention * V
    final_attention(P, V, O);

}


void krnl_attention(const float16_t* input, float16_t* output) {

    #pragma HLS INTERFACE mode=m_axi port=input depth=INPUT_SIZE bundle=gmem0
    #pragma HLS INTERFACE mode=m_axi port=output depth=OUTPUT_SIZE bundle=gmem1

    // Local buffers
    float Q[B][T][C];
    #pragma HLS ARRAY_PARTITION variable=Q type=block factor=INTERFACE_SIZE dim=3
    float K[B][T][C];
    #pragma HLS ARRAY_PARTITION variable=K type=block factor=INTERFACE_SIZE dim=3
    float V[B][T][C];
    #pragma HLS ARRAY_PARTITION variable=V type=block factor=INTERFACE_SIZE dim=3

    float O[B][T][C];
    #pragma HLS ARRAY_PARTITION variable=O type=block factor=INTERFACE_SIZE dim=3

    // Load from interface to local buffers
    load_input(input, Q, OFFSET_Q * INPUT_SIZE_SINGLE);
    load_input(input, K, OFFSET_K * INPUT_SIZE_SINGLE);
    load_input(input, V, OFFSET_V * INPUT_SIZE_SINGLE);

    // Attention algorithm
    attention(Q, K, V, O);

    // Store from local buffer to interface
    store_output(output, O);
    
}