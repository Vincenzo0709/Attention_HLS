# HLS Single-Head Attention
This is an HLS implementation of Single-Headed Attention algorithm:
- Input is [Q,K,V] concatenated on the same interface port;
- Output is on the other interface port;
- Both are in the same interface bundle;
- Each computation exploits parallelism on m_axi_port_t lines:
    - Enforcing __pipelining__ (II=1) between different lines;
    - Enforcing __unrolling__ into each line.
- The last multiplication by Values id optimized with a scalar-vector multiplication, between P elements and V rows:
    - To avoid inefficient column accesses for V tensor.
- Input rows are buffered in __local storages (BRAM)__ to reuse data and reduce DDR access latency;
- __Full array partitioning__ on inputs local storages for every elaboration;

>NOTE: P is a local storage for now, but it depends on (B,T,C) values. Furthermore it could be implemented as URAM, depending on dimensions and device.

>NOTE: how to partition (complete, cyclic or block) depends on input size and must be discussed.
