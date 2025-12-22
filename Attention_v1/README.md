# HLS Attention
This is an HLS implementation of Single-Headed Attention algorithm:
- Input is [Q,K,V] concatenated on the same interface port;
- Output is on the other interface port;
- Both are in the same interface bundle;
- Zero-copy approach on input/output access;
- Each computation exploits parallelism on m_axi_port_t lines:
    - Trying pipelining between different lines;
    - Trying unrolling into each line.
- The last multiplication by Values id optimized with a scalar-vector multiplication, between P elements and V rows:
    - To avoid inefficient column accesses for V tensor.
