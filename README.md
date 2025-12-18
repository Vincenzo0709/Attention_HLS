# HLS Attention
This repository holds some Single-Head Attention HLS implementations.
- Attention_v0: base version without any optimizations;
- 

# Compile
```
cd <version_dir>
make clean
make <target>
```

Where \<target\> can be:
- csim: C simulation;
- syn: RTL synthesis;
- cosim: RTL cosimulation with XSIM, after synthesis;
- package: IP packaging for Vivado.