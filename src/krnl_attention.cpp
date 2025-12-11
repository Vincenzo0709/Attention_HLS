#include "krnl_attention.h"

void load_input(const float* in, float mat, const int offset) {
    
    for(int i=0; i<B; i++) {

        for(int j=0; j<T; j++) {

            for(int k=0; k<C; k++) {

                sd
            
            }

        }

    }

}

void krnl_attention(const float* input, float* output) {

    #pragma HLS INTERFACE m_axi port=input depth=B*T*3*C bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output depth=B*T*3*C bundle=gmem1

    float Q[B][T][C];
    float K[B][T][C];
    float V[B][T][C];


    
}