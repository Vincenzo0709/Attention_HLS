#include "krnl_attention.h"

void load_input(const float* in, float mat[B][T][C], const int offset)
{
    
    for(int b=0; b<B; b++) {

        for(int t=0; t<T; t++) {

            for(int c=0; c<C; c++) {

                // The current index points to:
                // - b sequences ahead (sized T*3*C);
                // - t tokens ahead (sized 3*C);
                // - the offset is because of [Q,K,V] chained input;
                // - c elements ahead.
                int idx=b*T*3*C + t*3*C + offset*C + c;
                
                mat[b][t][c] = in[idx];
            
            }

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


void krnl_attention(const float* input, float* output)
{

    #pragma HLS INTERFACE m_axi port=input depth=B*T*3*C bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output depth=B*T*C bundle=gmem1

    // Local buffers
    float Q[B][T][C];
    float K[B][T][C];
    float V[B][T][C];

    float O[B][T][C];

    // Load from interface to local buffers
    load_input(input, Q, 0);
    load_input(input, K, 1);
    load_input(input, V, 2);

    // Attention algorithm
    attention(Q, K, V, O);

    // Store from local buffer to interface
    store_output(output, O);
    
}