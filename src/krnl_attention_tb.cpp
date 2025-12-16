#include <iostream>
#include "krnl_attention.h"
using namespace std;

// Software model to verify
void attention_sw(const float* input, float* output) {
    
    float (*Q)[T][C] = (float (*)[T][C]) malloc(B * T * C * sizeof(float));
    float (*K)[T][C] = (float (*)[T][C]) malloc(B * T * C * sizeof(float));
    float (*V)[T][C] = (float (*)[T][C]) malloc(B * T * C * sizeof(float));
    float (*P)[T][T] = (float (*)[T][T]) malloc(B * T * T * sizeof(float));

    // Load input
    for(int b=0; b<B; b++) {
        for(int t=0; t<T; t++) {
            for(int c=0; c<C; c++) {
                int idx_base = b*T*3*C + t*3*C;
                Q[b][t][c] = input[idx_base + 0*C + c];
                K[b][t][c] = input[idx_base + 1*C + c];
                V[b][t][c] = input[idx_base + 2*C + c];
            }
        }
    }

    float scale = 1.0f / sqrtf((float)C);

    // Attention
    for(int b=0; b<B; b++) {
        for(int t=0; t<T; t++) {

            // QK^T
            for(int t2=0; t2<=t; t2++) {
                float sum = 0.0f;
                for(int c=0; c<C; c++) sum += Q[b][t][c] * K[b][t2][c];
                P[b][t][t2] = sum * scale;
            }

            // Softmax
            float max = -1e10f;
            for(int t2=0; t2<=t; t2++) if(P[b][t][t2] > max) max = P[b][t][t2];
            
            float expsum = 0.0f;
            for(int t2=0; t2<=t; t2++) {
                float e = expf(P[b][t][t2] - max);
                P[b][t][t2] = e;
                expsum += e;
            }

            for(int t2=0; t2<=t; t2++) P[b][t][t2] /= expsum;

            // Attention * V
            for(int c=0; c<C; c++) {
                float sum = 0.0f;
                for(int t2=0; t2<=t; t2++) sum += P[b][t][t2] * V[b][t2][c];
                
                // Store output
                output[b*T*C + t*C + c] = sum;
            }
        }
    }

    free(Q);
    free(K);
    free(V);
    free(P);
}


int main() {
    printf("\n--- Starting Attention testbench ---\n");

    // Sizes
    int size_input = B*T*3*C;
    int size_output = B*T*C;

    printf("Dimensions: B=%d, T=%d, C=%d\n", B, T, C);

    // Allocazione Memoria
    float *input      = (float*)malloc(size_input * sizeof(float));
    float *output_hls = (float*)malloc(size_output * sizeof(float));
    float *output_sw  = (float*)malloc(size_output * sizeof(float));

    // Input data initialization (random values between -1.0 and 1.0)
    for(int i=0; i<size_input; i++) {
        input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    // Software model execution
    printf("Software model execution (CPU)...\n");
    attention_sw(input, output_sw);

    // HLS kernel execution
    printf("HLS kernel execution...\n");
    krnl_attention(input, output_hls);

    // Confronting
    printf("Result verification...\n");
    int errors = 0;
    float max_diff = 0.0f;
    float epsilon = 1e-2;

    for(int i=0; i<size_output; i++) {
        float diff = fabs(output_hls[i] - output_sw[i]);
        if(diff > max_diff) max_diff = diff;

        if(diff > epsilon) {
            errors++;
            if (errors < 10) {
                // Printing the first 10 errors
                printf("Error at index %d: HLS=%f, SW=%f, Diff=%f\n", 
                        i, output_hls[i], output_sw[i], diff);
            }
        }
    }

    // Report
    if(errors == 0) {
        printf("SUCCESS!\n\n");
        printf("Maximum diff: %e\n", max_diff);
    } else {
        printf("\nTEST failed! %d errors found.\n", errors);
    }

    // Cleanup
    free(input);
    free(output_hls);
    free(output_sw);

    return (errors == 0) ? 0 : 1;
}