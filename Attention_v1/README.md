# HLS Single-Head Attention
This is an HLS implementation of Single-Headed Attention algorithm:
- Input is [Q,K,V] concatenated on the same interface port;
- Output is on the other interface port;
- Both are in the same interface bundle;
- __Zero-copy__ approach on input/output access from/to DDR;
- Each computation exploits parallelism on m_axi_port_t lines:
    - Trying __pipelining__ (II=1) between different lines;
    - Trying __unrolling__ into each line.
- The last multiplication by Values is optimized with a __scalar-vector multiplication__, between P elements and V rows:
    - To avoid inefficient column accesses for V tensor.
