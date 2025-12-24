BUILD_DIR := ./builds/
SRC_DIR := ./src/
MES_DIR := ./time_measurements/
GRAPHS_DIR := ${MES_DIR}graphs/

all: global shared

global: ${SRC_DIR}convolution.cu
	nvcc ${SRC_DIR}convolution.cu -o ${BUILD_DIR}global

shared: ${SRC_DIR}shared_convolution.cu
	nvcc ${SRC_DIR}shared_convolution.cu -o ${BUILD_DIR}shared

run: run_shared run_global

run_shared:
	${BUILD_DIR}shared

run_global:
	${BUILD_DIR}global

clean:
	rm -rf ${BUILD_DIR}* ${GRAPHS_DIR}* ${MES_DIR}*.dat


