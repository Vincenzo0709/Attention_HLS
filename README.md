# HLS Attention
This repository holds some Single-Head Attention HLS implementations.
- Attention_v0: base version without any optimizations;
- Attention_v1: 

# Compile
```
cd <version_dir>
```

You can choose between three data types:
- float16;
- float32;
- double.
by adding into Makefile the line:
```
CPPFLAGS = -D<type>
```

Then, to compile:
```
make clean
make <target>
```

Where \<target\> can be:
- csim: C simulation;
- syn: RTL synthesis;
- cosim: RTL cosimulation with XSIM, after synthesis;
- package: IP packaging for Vivado.