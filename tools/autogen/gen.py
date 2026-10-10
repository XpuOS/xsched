#!/usr/bin/env python3
"""
Code generator for driver and intercept files using Clang.
This script parses C/C++ header files and generates corresponding driver and intercept files.
"""

import os
import sys
import site
import pathlib
import argparse
import subprocess
import platform as sys_platform
from typing import List, Optional
import clang.cindex

# please install libclang first: pip install libclang
# recursively find libclang under site-packages
libclang_path = None
libclang_pattern = 'libclang.so' if sys_platform.system() != 'Windows' else 'libclang.dll'

for site_package in site.getsitepackages():
    for root, dirs, files in os.walk(site_package):
        for file in files:
            if file.startswith(libclang_pattern):
                libclang_path = os.path.join(root, file)
                break
        if libclang_path:
            break

if not libclang_path:
    raise FileNotFoundError(f"Could not find {libclang_pattern} in site-packages")

print(f"Using libclang: {libclang_path}")
clang.cindex.Config.set_library_file(libclang_path)

# Hand-written interception targets.
# platforms/cuda/shim/src/intercept.cpp redirects these symbols to the X*
# implementations (defined in platforms/cuda/shim/src/shim.cpp) instead of the
# default Driver:: passthrough. The list must stay in sync with the actual
# hooks in intercept.cpp; regenerating without it would silently revert every
# hook to a plain passthrough (losing XSched's scheduling/synchronization
# logic). The redirect target follows the naming rule "X" + symbol[prefix_len:].
HOOKED_SYMBOLS = {
    # context / stream lifecycle and synchronization
    "cuCtxSynchronize", "cuCtxDestroy", "cuCtxDestroy_v2",
    "cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2",
    "cuDevicePrimaryCtxReset", "cuDevicePrimaryCtxReset_v2",
    "cuStreamCreate", "cuStreamCreateWithPriority",
    "cuStreamDestroy", "cuStreamDestroy_v2",
    "cuStreamSynchronize", "cuStreamSynchronize_ptsz",
    "cuStreamQuery", "cuStreamQuery_ptsz",
    # kernel launches (incl. ptsz variants and cooperative kernels)
    "cuLaunchKernel", "cuLaunchKernel_ptsz",
    "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz",
    "cuLaunchCooperativeKernel", "cuLaunchCooperativeKernel_ptsz",
    "cuLaunchHostFunc", "cuLaunchHostFunc_ptsz",
    # events
    "cuEventRecord", "cuEventRecord_ptsz",
    "cuEventRecordWithFlags", "cuEventRecordWithFlags_ptsz",
    "cuEventQuery", "cuEventSynchronize",
    "cuEventDestroy", "cuEventDestroy_v2",
    "cuEventElapsedTime", "cuEventElapsedTime_v2",
    # memory
    "cuMemFree_v2", "cuMemFreeAsync", "cuMemFreeAsync_ptsz",
    "cuMemAllocAsync", "cuMemAllocAsync_ptsz",
    "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuMemcpyDtoD_v2",
    "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoDAsync_v2",
    "cuMemcpy2DAsync_v2", "cuMemcpy3DAsync_v2",
    "cuMemcpyHtoD_v2_ptds", "cuMemcpyDtoH_v2_ptds", "cuMemcpyDtoD_v2_ptds",
    "cuMemcpyHtoDAsync_v2_ptsz", "cuMemcpyDtoHAsync_v2_ptsz", "cuMemcpyDtoDAsync_v2_ptsz",
    "cuMemcpy2DAsync_v2_ptsz", "cuMemcpy3DAsync_v2_ptsz",
    "cuMemsetD8Async", "cuMemsetD16Async", "cuMemsetD32Async",
    "cuMemsetD2D8Async", "cuMemsetD2D16Async", "cuMemsetD2D32Async",
    "cuMemsetD8Async_ptsz", "cuMemsetD16Async_ptsz", "cuMemsetD32Async_ptsz",
    "cuMemsetD2D8Async_ptsz", "cuMemsetD2D16Async_ptsz", "cuMemsetD2D32Async_ptsz",
    # stream capture / graph launch
    "cuStreamBeginCapture", "cuStreamBeginCapture_ptsz",
    "cuStreamBeginCapture_v2", "cuStreamBeginCapture_v2_ptsz",
    "cuStreamBeginCaptureToGraph", "cuStreamBeginCaptureToGraph_ptsz",
    "cuStreamEndCapture", "cuStreamEndCapture_ptsz",
    "cuStreamWaitEvent", "cuStreamWaitEvent_ptsz",
    "cuGraphUpload", "cuGraphUpload_ptsz",
    "cuGraphLaunch", "cuGraphLaunch_ptsz",
    # proc address
    "cuGetProcAddress", "cuGetProcAddress_v2",
}


def hook_target(function_name: str, prefix: str) -> str:
    """Redirect hooked symbols to their X* implementation, others to Driver::."""
    if function_name in HOOKED_SYMBOLS:
        return f"X{function_name[len(prefix):]}"
    return f"Driver::{function_name[len(prefix):]}"

class TypeGenerator:
    """Handles generation of typedefs for complex types."""
    
    @staticmethod
    def is_complex_type(type_obj: clang.cindex.Type) -> bool:
        """Check if the type is a function pointer or array."""
        return (type_obj.kind == clang.cindex.TypeKind.POINTER and 
                type_obj.get_pointee().kind == clang.cindex.TypeKind.FUNCTIONPROTO) or \
               type_obj.kind == clang.cindex.TypeKind.CONSTANTARRAY or \
               type_obj.kind == clang.cindex.TypeKind.INCOMPLETEARRAY

    @staticmethod
    def generate_typedef_name(func_name: str, param_index: int) -> str:
        """Generate typedef name for complex types."""
        return f"{func_name}_arg{param_index}_t"

    @staticmethod
    def generate_typedef(type_obj: clang.cindex.Type, typedef_name: str) -> str:
        """Generate typedef string for complex types."""
        if type_obj.kind == clang.cindex.TypeKind.POINTER:
            pointee = type_obj.get_pointee()
            if pointee.kind == clang.cindex.TypeKind.FUNCTIONPROTO:
                # Function pointer
                return_type = pointee.get_result().spelling
                param_types = [param.spelling for param in pointee.argument_types()]
                param_list = ", ".join(param_types)
                return f"typedef {return_type} (*{typedef_name})({param_list});"
        elif type_obj.kind in [clang.cindex.TypeKind.CONSTANTARRAY, clang.cindex.TypeKind.INCOMPLETEARRAY]:
            # Array type
            element_type = type_obj.element_type.spelling
            if type_obj.kind == clang.cindex.TypeKind.INCOMPLETEARRAY:
                return f"typedef {element_type} {typedef_name}[];"
            else:
                array_size = type_obj.element_count
                return f"typedef {element_type} {typedef_name}[{array_size}];"
        return ""


class CodeGenerator:
    """Main class for generating driver and intercept code."""
    
    def __init__(self, platform: str, prefix: str, lib_name: str, lib_dir: str):
        self.platform = platform
        self.prefix = prefix
        self.prefix_len = len(prefix)
        self.lib_name = lib_name
        self.lib_dir = lib_dir
        self.driver_symbols: List[str] = []
        self.typedefs_str = ""
        self.driver_str = ""
        self.intercept_str = ""
        self.intercept_entry_str = ""

    def get_library_symbols(self, lib_path: pathlib.Path) -> None:
        """Extract symbols from the library file using nm or dumpbin."""
        if not lib_path.exists():
            raise FileNotFoundError(f"Library file {lib_path} does not exist")

        if sys_platform.system() == 'Windows':
            # Use dumpbin on Windows
            cmd = f"dumpbin /EXPORTS {lib_path}"
            try:
                output = subprocess.check_output(cmd, shell=True).decode()
                # Parse output to get exported symbols
                lines = output.splitlines()
                start_parsing = False
                for line in lines:
                    if "ordinal" in line and "name" in line:
                        start_parsing = True
                        continue
                    if start_parsing and line.strip() == "":
                        continue
                    if "Summary" == line:
                        start_parsing = False
                        continue
                    if start_parsing:
                        parts = line.split()
                        if len(parts) >= 4:
                            self.driver_symbols.append(parts[3])
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to run dumpbin: {e}")
        else:
            # Use nm on Linux/Unix
            cmd = f"nm -D {lib_path} | awk '$2 == \"T\" {{sub(/@.*/, \"\", $3); print $3}}'"
            self.driver_symbols = subprocess.check_output(cmd, shell=True).decode().splitlines()
        
        print(f"Found {len(self.driver_symbols)} symbols in {lib_path}")

    def parse_function(self, function_name: str, return_type: str, parameters: List[clang.cindex.Cursor]) -> None:
        """Parse a function declaration and generate corresponding code."""
        params = list(parameters)
        self.intercept_entry_str += f"    DLSYM_INTERCEPT_ENTRY({function_name}),\n"
        
        # Generate typedefs for complex types
        param_types = []
        for i, param in enumerate(params):
            if TypeGenerator.is_complex_type(param.type):
                typedef_name = TypeGenerator.generate_typedef_name(function_name, i)
                typedef_str = TypeGenerator.generate_typedef(param.type, typedef_name)
                if typedef_str:
                    self.typedefs_str += typedef_str + "\n"
                    param_types.append(f"{typedef_name}, {param.spelling}")
            else:
                param_types.append(f"{param.type.spelling}, {param.spelling}")

        if len(params) == 0:
            self.driver_str += f"    DEFINE_STATIC_ADDRESS_CALL(GetSymbol(\"{function_name}\"), {return_type}, {function_name[self.prefix_len:]});\n"
            self.intercept_str += f"DEFINE_EXPORT_C_REDIRECT_CALL({hook_target(function_name, self.prefix)}, {return_type}, {function_name});\n"
            return

        self.driver_str += f"    DEFINE_STATIC_ADDRESS_CALL(GetSymbol(\"{function_name}\"), {return_type}, {function_name[self.prefix_len:]}, {', '.join(param_types)});\n"
        self.intercept_str += f"DEFINE_EXPORT_C_REDIRECT_CALL({hook_target(function_name, self.prefix)}, {return_type}, {function_name}, {', '.join(param_types)});\n"

    def find_functions(self, node: clang.cindex.Cursor) -> None:
        """Recursively find and process function declarations."""
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            # Skip inline functions
            if node.is_definition() and "inline" in [token.spelling for token in node.get_tokens()]:
                print(f"Skipping inline function: {node.spelling}")
                return

            function_name = node.spelling
            # for Ascend, filter out functions in libopapi.so that end with GetWorkspaceSize
            # if function_name.endswith("GetWorkspaceSize"):
            #     return
            if function_name in self.driver_symbols and function_name.startswith(self.prefix):
                return_type = node.result_type.spelling
                parameters = node.get_arguments()
                self.parse_function(function_name, return_type, parameters)

        for child in node.get_children():
            self.find_functions(child)

    def parse_source_file(self, file_path: pathlib.Path, include_paths: Optional[List[str]] = None) -> None:
        """Parse the source file and generate code."""
        index = clang.cindex.Index.create()
        args = ['-x', 'c++']  # Force C++ parsing
        if include_paths:
            for path in include_paths:
                args.append(f'-I{path}')
        translation_unit = index.parse(str(file_path), args=args)
        self.find_functions(translation_unit.cursor)

    def write_output_files(self, driver_file: str, intercept_file: str, cmd: str, source_name: str) -> None:
        """Write the generated code to output files."""
        if self.typedefs_str:
            self.typedefs_str = "\n" + self.typedefs_str

        with open(driver_file, "w", newline='\n') as f:
            f.write(f"""/// This file is auto-generated by command \"{cmd}\"
#pragma once

#include "xsched/protocol/def.h"
#include "xsched/utils/common.h"
#include "xsched/utils/symbol.h"
#include "xsched/utils/function.h"
#include "xsched/{self.platform}/hal/{source_name}"

namespace xsched::{self.platform}
{{
{self.typedefs_str}
class Driver
{{
private:
    /// FIXME
    DEFINE_GET_SYMBOL_FUNC(GetSymbol, "XSCHED_{self.platform.upper()}_LIB",
                           std::vector<std::string>({{"{self.lib_name}"}}), // search name
                           std::vector<std::string>({{"{self.lib_dir}"}})); // search path

public:
    STATIC_CLASS(Driver);

{self.driver_str}
}};

}} // namespace xsched::{self.platform}
""")

        intercept_lib_str = ""
        if sys_platform.system() == 'Windows':
            intercept_lib_str = "DEFINE_DLSYM_INTERCEPT(intercept_symbol_map);"
        else:
            intercept_lib_str = f'DLSYM_INTERCEPT_LIB(intercept_libs,\n\t"{self.lib_name}"\n);\nDEFINE_DLSYM_INTERCEPT(intercept_symbol_map, intercept_libs);'
        with open(intercept_file, "w", newline='\n') as f:
            f.write(f"""/// This file is auto-generated by command \"{cmd}\"
#include "xsched/utils/function.h"
#include "xsched/{self.platform}/hal/driver.h"

using namespace xsched::{self.platform};

{self.intercept_str}
static const std::unordered_map<std::string, void *> intercept_symbol_map = {{
{self.intercept_entry_str}}};

{intercept_lib_str}
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate driver and intercept files for a given header file.')
    parser.add_argument('-s', '--source', type=str, required=True, help='Path to the header source')
    parser.add_argument('-i', '--include', type=str, action='append', help='Path to the include directory')
    parser.add_argument('--platform', type=str, required=True, help='Platform name')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for the function names')
    parser.add_argument('--lib', type=str, required=True, help='Driver library file')
    parser.add_argument('--driver', type=str, default="driver.h", help='Output driver header file')
    parser.add_argument('--intercept', type=str, default="intercept.cpp", help='Output intercept source file')
    args = parser.parse_args()

    # Construct the command string for documentation
    cmd = "python3 " + " ".join(sys.argv)

    # Initialize paths
    source_path = pathlib.Path(args.source)
    lib_path = pathlib.Path(args.lib).absolute()
    source_name = source_path.name

    # Create code generator and process files
    generator = CodeGenerator(args.platform, args.prefix, lib_path.name, str(lib_path.parent))
    generator.get_library_symbols(lib_path)
    generator.parse_source_file(source_path, args.include)
    generator.write_output_files(args.driver, args.intercept, cmd, source_name)
