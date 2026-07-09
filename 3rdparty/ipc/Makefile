# BUILD_TYPE		= Debug / Release
BUILD_TYPE			= Release
# VERBOSE			= ON / OFF : enable verbose makefile
VERBOSE				= OFF
# BUILD_TEST		= ON / OFF : enable build test
BUILD_TEST			= ON
# WINDOWS_COMPILER  = MSVC / MINGW
WINDOWS_COMPILER 	= MINGW

ifeq ($(OS),Windows_NT)
    WORK_PATH = $(CURDIR)
	RM_CMD = @if exist "$(1)" rmdir /s /q "$(1)"
	CPU_CORES = $(NUMBER_OF_PROCESSORS)
    ifeq ($(WINDOWS_COMPILER), MSVC)
        CMAKE_GENERATOR = 
        CMAKE_CXX_COMPILER = 
		PARALLEL_OPTIONS = -- /m:$(CPU_CORES)
    else ifeq ($(WINDOWS_COMPILER), MINGW)
        CMAKE_GENERATOR = -G "MinGW Makefiles"
        CMAKE_CXX_COMPILER = -DCMAKE_CXX_COMPILER=g++
		PARALLEL_OPTIONS = -- -j$(CPU_CORES)
    endif
else
    WORK_PATH = $(shell pwd)
	RM_CMD = rm -rf $(1)
	CPU_CORES = $(shell nproc)
	CMAKE_GENERATOR = 
    CMAKE_CXX_COMPILER = 
	PARALLEL_OPTIONS = -- -j$(CPU_CORES)
endif

BUILD_PATH		= ${WORK_PATH}/build
OUTPUT_PATH		= ${WORK_PATH}/output

.PHONY: build
build: ${BUILD_PATH}/CMakeCache.txt
	$(call RM_CMD,${OUTPUT_PATH})
	cmake --build ${BUILD_PATH} --target install $(PARALLEL_OPTIONS)

${BUILD_PATH}/CMakeCache.txt:
	${MAKE} configure

.PHONY: configure
configure:
	cmake -B${BUILD_PATH}							\
		  ${CMAKE_GENERATOR}				        \
		  ${CMAKE_CXX_COMPILER}						\
		  -DCMAKE_BUILD_TYPE=${BUILD_TYPE}			\
		  -DCMAKE_VERBOSE_MAKEFILE=${VERBOSE}		\
		  -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}  	\
		  -DBUILD_TEST=${BUILD_TEST}

.PHONY: clean
clean:
	$(call RM_CMD,${BUILD_PATH})
	$(call RM_CMD,${OUTPUT_PATH})
