# BUILD_TYPE		= Debug / Release
BUILD_TYPE			= Release

# VERBOSE			= ON / OFF : enable verbose makefile
VERBOSE				= OFF

# SHIM_SOFTLINK		= ON / OFF : create softlink for shim library on Linux
SHIM_SOFTLINK		= ON

# BUILD_TEST		= ON / OFF : build test cases
BUILD_TEST			= OFF

# PLATFORM			= NONE / ascend / cuda / cudla / hip / levelzero / opencl / vpi / template
PLATFORM			= NONE

ifeq (${OS}, Windows_NT)
	WORK_PATH		= ${CURDIR}
	CPU_CORES		= ${NUMBER_OF_PROCESSORS}
	CMAKE_OS_FLAGS	= -G "MinGW Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
	rm_dir			= @if exist "${1}" rmdir /s /q "${1}"
else
	WORK_PATH		= $(dir $(abspath $(lastword ${MAKEFILE_LIST})))
	CPU_CORES		= $(shell nproc)
	CMAKE_OS_FLAGS	=
	rm_dir			= rm -rf ${1}
endif

BUILD_PATH			= ${WORK_PATH}/build
INSTALL_PATH		= ${WORK_PATH}/output
LIB_PATH			= ${INSTALL_PATH}/lib

define uppercase
$(subst z,Z,$(subst y,Y,$(subst x,X,$(subst w,W,$(subst v,V,$(subst u,U,$(subst t,T,$(subst s,S,$(subst r,R,$(subst q,Q,$(subst p,P,$(subst o,O,$(subst n,N,$(subst m,M,$(subst l,L,$(subst k,K,$(subst j,J,$(subst i,I,$(subst h,H,$(subst g,G,$(subst f,F,$(subst e,E,$(subst d,D,$(subst c,C,$(subst b,B,$(subst a,A,$(1)))))))))))))))))))))))))))
endef

.PHONY: build
build: ${BUILD_PATH}/CMakeCache.txt
	$(call rm_dir,${INSTALL_PATH})
	cmake --build ${BUILD_PATH} --target install -- -j${CPU_CORES}

${BUILD_PATH}/CMakeCache.txt:
	${MAKE} configure

.PHONY: configure
configure:
	cmake -B${BUILD_PATH}									\
		  -DCMAKE_BUILD_TYPE=${BUILD_TYPE}					\
		  -DCMAKE_VERBOSE_MAKEFILE=${VERBOSE}				\
		  -DCMAKE_INSTALL_PREFIX=$(abspath $(INSTALL_PATH))	\
		  -DSHIM_SOFTLINK=${SHIM_SOFTLINK}					\
		  -DBUILD_TEST=${BUILD_TEST}						\
		  ${CMAKE_OS_FLAGS}									\
		  $(foreach p,$(PLATFORM),-DPLATFORM_$(call uppercase,${p})=ON)

.PHONY: clean
clean:
	$(call rm_dir,${BUILD_PATH})
	$(call rm_dir,${INSTALL_PATH})

.PHONY: ascend
ascend:
	${MAKE} clean
	${MAKE} PLATFORM=ascend

.PHONY: cuda
cuda:
	${MAKE} clean
	${MAKE} PLATFORM=cuda

.PHONY: cudla
cudla:
	${MAKE} clean
	${MAKE} PLATFORM=cudla

.PHONY: hip
hip:
	${MAKE} clean
	${MAKE} PLATFORM=hip

.PHONY: levelzero
levelzero:
	${MAKE} clean
	${MAKE} PLATFORM=levelzero

.PHONY: opencl
opencl:
	${MAKE} clean
	${MAKE} PLATFORM=opencl

.PHONY: vpi
vpi:
	${MAKE} clean
	${MAKE} PLATFORM=vpi

.PHONY: template
template:
	${MAKE} clean
	${MAKE} PLATFORM=template
