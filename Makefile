# Author: Vincenzo Merola <vincenzo.merola2@unina.it>
# Description:
#   	This Makefile uses Vitis compiler to simulate and synthesize HLS kernel.

DIR = attention

VITIS_HLS = vitis-run --mode hls
VPP = v++
VFLAGS = -c --mode hls
CSIM = ${VITIS_HLS} --csim
COSIM = ${VITIS_HLS} --cosim
SYN = ${VPP} ${VFLAGS}
PACK = ${VITIS_HLS} --package
IP = ${DIR}_hls

CONFIG = --config src/hls_config.cfg
WORK_DIR = --work_dir ${DIR}


csim:
	@echo
	@echo "C-Simulation starting..."
	${CSIM} ${CONFIG} ${WORK_DIR}
	@echo "C-Simulation ended"
	@echo


syn:
	@echo
	@echo "Synthesis starting..."
	${SYN} ${CONFIG} ${WORK_DIR}
	@echo "Synthesis ended"
	@echo


cosim: syn
	@echo
	@echo "Cosimulation starting ..."
	${COSIM} ${CONFIG} ${WORK_DIR}
	@echo "Cosimulation ended"
	@echo


package: syn
	@echo
	@echo "Packaging IP..."
	${PACK} ${CONFIG} ${WORK_DIR}
	@echo "Packaging ended"
	@echo


clean:
	@echo "Cleaning..."
	rm -f ${IP}.zip

	rm -rf ${DIR}
	rm -f xcd.log
	rm -rf .Xil

.PHONY: csim syn cosim package clean
