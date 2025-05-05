# BUILD_TYPE		= Debug / Release
BUILD_TYPE			= Release

# VERBOSE			= ON / OFF : enable verbose makefile
VERBOSE				= OFF

SHIM_SOFTLINK		= ON
CONTAINER_SUPPORT	= OFF
BUILD_TEST			= OFF

PLATFORM			= NONE
WORK_PATH			= $(shell pwd)
TEST_PATH			= ${WORK_PATH}/test
BUILD_PATH			= ${WORK_PATH}/build
OUTPUT_PATH			= ${WORK_PATH}/output
LIB_PATH			= ${OUTPUT_PATH}/lib

.PHONY: build
build: ${BUILD_PATH}/CMakeCache.txt
	rm -rf ${OUTPUT_PATH}; \
	cmake --build ${BUILD_PATH} --target install -- -j$(shell nproc)

${BUILD_PATH}/CMakeCache.txt:
	${MAKE} configure

.PHONY: configure
configure:
	cmake -B${BUILD_PATH}	\
		  -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}		\
		  -DCMAKE_BUILD_TYPE=${BUILD_TYPE}			\
		  -DCMAKE_VERBOSE_MAKEFILE=${VERBOSE}		\
		  -DPLATFORM_$(shell echo ${PLATFORM} | tr '[:lower:]' '[:upper:]')=ON \
		  -DSHIM_SOFTLINK=${SHIM_SOFTLINK}			\
		  -DCONTAINER_SUPPORT=${CONTAINER_SUPPORT}	\
		  -DBUILD_TEST=${BUILD_TEST}

.PHONY: clean
clean:
	@rm -rf ${BUILD_PATH} ${OUTPUT_PATH}

.PHONY: ascend
ascend:
	${MAKE} clean; \
	${MAKE} PLATFORM=ascend

.PHONY: cuda
cuda:
	${MAKE} clean; \
	${MAKE} PLATFORM=cuda

.PHONY: cudla
cudla:
	${MAKE} clean; \
	${MAKE} PLATFORM=cudla

.PHONY: hip
hip:
	${MAKE} clean; \
	${MAKE} PLATFORM=hip

.PHONY: levelzero
levelzero:
	${MAKE} clean; \
	${MAKE} PLATFORM=levelzero

.PHONY: opencl
opencl:
	${MAKE} clean; \
	${MAKE} PLATFORM=opencl

.PHONY: template
template:
	${MAKE} clean; \
	${MAKE} PLATFORM=template

.PHONY: vpi
vpi:
	${MAKE} clean; \
	${MAKE} PLATFORM=vpi
