#include <iostream>
#include <chrono>
#include <vector>
#include "attention_func.h"
using namespace std;

// Helper function to read from m_axi_port_t
target_type_t read_vec(const m_axi_port_t* buffer, int global_idx) {
    int line_idx = global_idx / INTERFACE_SIZE;
    int elem_idx = global_idx % INTERFACE_SIZE;
    return buffer[line_idx][elem_idx];
}

// Helper function to write to m_axi_port_t
void write_vec(m_axi_port_t* buffer, int global_idx, target_type_t val) {
    int line_idx = global_idx / INTERFACE_SIZE;
    int elem_idx = global_idx % INTERFACE_SIZE;
    buffer[line_idx][elem_idx] = val;
}

// Software model to verify
void attention_sw(
                    const m_axi_port_t* input,
                    m_axi_port_t* output
                ) {

    const m_axi_port_t *Q_ptr = input + OFFSET_Q;
    const m_axi_port_t *K_ptr = input + OFFSET_K;
    const m_axi_port_t *V_ptr = input + OFFSET_V;

    m_axi_port_t P[B*T*T / INTERFACE_SIZE] = {0};
    m_axi_port_t O[OUTPUT_LINES];

    target_type_t scale = 1.0 / sqrtf(C);

    // Attention
    for(int b=0; b<B; b++) {
        for(int t=0; t<T; t++) {

            // QK^T
            for(int t2=0; t2<=t; t2++) {
                target_type_t sum = 0.0f;
                for(int c=0; c<C; c++) {
                    int q_idx = b*T*C + t*C + c;
                    int k_idx = b*T*C + t2*C + c;
                    sum += read_vec(Q_ptr, q_idx) * read_vec(K_ptr, k_idx);
                }

                int p_idx = b*T*T + t*T + t2;
                write_vec(P, p_idx, sum*scale);
                
            }

            // Softmax
            target_type_t max = -1e10;
            for(int t2=0; t2<=t; t2++) {
                int p_idx = b*T*T + t*T + t2;
                target_type_t val = read_vec(P, p_idx);
                if(val > max) max = val;
            }

            target_type_t expsum = 0.0;
            for(int t2=0; t2<=t; t2++) {
                int p_idx = b*T*T + t*T + t2;
                target_type_t val = read_vec(P, p_idx);
                target_type_t e = expf(val - max);
                
                write_vec(P, p_idx, e);
                expsum += e;
            }

            for(int t2=0; t2<=t; t2++) {
                int p_idx = b*T*T + t*T + t2;
                target_type_t val = read_vec(P, p_idx);
                write_vec(P, p_idx, val / expsum);
            }

            // Attention * V
            for(int c=0; c<C; c++) {
                target_type_t sum = 0.0f;
                for(int t2=0; t2<=t; t2++) {
                    int p_idx = b*T*T + t*T + t2;
                    int v_idx = b*T*C + t2*C + c;
                    
                    sum += read_vec(P, p_idx) * read_vec(V_ptr, v_idx);
                }
                int o_idx = b*T*C + t*C + c;
                write_vec(O, o_idx, sum);
            }
        }
    }

    for (int i=0; i<OUTPUT_LINES; i++) {
        output[i] = O[i];
    }
}

int main() {
    cout << "--- Starting Attention testbench ---" << endl;

    cout << "Dimensions: B=" << B << ", T=" << T << ", C=" << C << endl;

    // Allocazione Memoria
    m_axi_port_t input[INPUT_LINES];
    m_axi_port_t output_hls[OUTPUT_LINES];
    m_axi_port_t output_sw[OUTPUT_LINES];

    // Input data initialization (random values between -1.0 and 1.0)
    for(int i=0; i<INPUT_LINES; i++) {
        for (int j=0; j<INTERFACE_SIZE; j++) {
            input[i][j] = ((target_type_t)rand() / (target_type_t)RAND_MAX) * 2.0f - 1.0f;
        }
    }

    // Software model execution
    cout << "Software model execution (CPU)..." << endl;
    attention_sw(input, output_sw);

    // HLS kernel execution
    cout << "HLS kernel execution..." << endl;
    auto start = chrono::high_resolution_clock::now();
    krnl_attention(input, output_hls);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    cout << "Tempo esecuzione kernel: " << diff.count() << " s" << endl;

    // Confronting
    cout << "Result verification..." << endl;
    int errors = 0;
    target_type_t max_diff = 0.0f;
    target_type_t epsilon = 1e-2;

    for(int i=0; i<OUTPUT_LINES; i++) {
        for (int j=0; j<INTERFACE_SIZE; j++) {
            target_type_t diff = fabs(output_hls[i][j] - output_sw[i][j]);
            if(diff > max_diff) max_diff = diff;

            if(diff > epsilon) {
                errors++;
                if (errors < 10) {
                    // Printing the first 10 errors
                    cout << "Error at index "<< i << ": HLS=" << output_hls[i][j]
                            << ", SW=" << output_sw[i][j] << ", Diff=" << diff << endl;
                }
            }
        }
    }

    // Report
    if(errors == 0) {
        cout << "SUCCESS!" << endl << endl;
        cout << "Maximum diff: "<< max_diff;
    } else {
        cout << "TEST failed! " << errors << " errors found." << endl;
    }

    return (errors == 0) ? 0 : 1;
}