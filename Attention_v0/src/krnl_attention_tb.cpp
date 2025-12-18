#include <iostream>
#include <chrono>
#include <vector>
#include "krnl_attention.h"
using namespace std;

// Software model to verify
void attention_sw(
                    const m_axi_port_t* input,
                    m_axi_port_t* output
                ) {
    
    target_type_t Q[B][T][C] = {0};
    target_type_t K[B][T][C] = {0};
    target_type_t V[B][T][C] = {0};
    target_type_t P[B][T][T] = {0};
    target_type_t O[B][T][C] = {0};

    int curr_line_Q = OFFSET_Q / INTERFACE_SIZE;
    int curr_line_K = OFFSET_K / INTERFACE_SIZE;
    int curr_line_V = OFFSET_V / INTERFACE_SIZE;

    // Load input
    for(int b=0; b<B; b++) {
        for(int t=0; t<T; t++) {
            for (int line=0; line<C/INTERFACE_SIZE; line++) {

                m_axi_port_t buff_Q = input[curr_line_Q];
                m_axi_port_t buff_K = input[curr_line_K];
                m_axi_port_t buff_V = input[curr_line_V];

                for(int c=0; c<INTERFACE_SIZE; c++) {
                    Q[b][t][c + line*INTERFACE_SIZE] = buff_Q[c];
                    K[b][t][c + line*INTERFACE_SIZE] = buff_K[c];
                    V[b][t][c + line*INTERFACE_SIZE] = buff_V[c];
                }

                curr_line_Q++;
                curr_line_K++;
                curr_line_V++;

            }
        }
    }

    target_type_t scale = 1.0 / sqrtf(C);

    // Attention
    for(int b=0; b<B; b++) {
        for(int t=0; t<T; t++) {

            // QK^T
            for(int t2=0; t2<=t; t2++) {
                target_type_t sum = 0.0f;
                for(int c=0; c<C; c++) sum += Q[b][t][c] * K[b][t2][c];
                P[b][t][t2] = sum * scale;
            }

            // Softmax
            target_type_t max = -1e10;
            for(int t2=0; t2<=t; t2++) if(P[b][t][t2] > max) max = P[b][t][t2];
            
            target_type_t expsum = 0.0;
            for(int t2=0; t2<=t; t2++) {
                target_type_t e = expf(P[b][t][t2] - max);
                P[b][t][t2] = e;
                expsum += e;
            }

            for(int t2=0; t2<=t; t2++) P[b][t][t2] /= expsum;

            // Attention * V
            for(int c=0; c<C; c++) {
                target_type_t sum = 0.0f;
                for(int t2=0; t2<=t; t2++) sum += P[b][t][t2] * V[b][t2][c];
                
                // Store output
                O[b][t][c] = sum;
            }
        }
    }

    int curr_line = 0;
    for(int b=0; b<B; b++) {
        for(int t=0; t<T; t++) {
            m_axi_port_t buff;
            for (int line=0; line<C/INTERFACE_SIZE; line++) {
                for(int c=0; c<INTERFACE_SIZE; c++) {
                    buff[c] = O[b][t][c + line*INTERFACE_SIZE];
                }
                output[curr_line] = buff;
                curr_line++;
            }
        }
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