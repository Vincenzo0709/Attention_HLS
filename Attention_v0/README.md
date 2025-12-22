# HLS Single-Head Attention
This is an HLS implementation of Single-Headed Attention algorithm:
- Input is [Q,K,V] concatenated on the same interface port;
- Output is on the other interface port;
- Both are in the same interface bundle;
- Tensors are used as 'target_type_t' matrices;
- There is no optimization in matmuls or data load/store.
