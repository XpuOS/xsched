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
from typing import List, Optional
import clang.cindex

# please install libclang first: pip install libclang
# recursively find libclang under site-packages
libclang_path = None
for site_package in site.getsitepackages():
    for root, dirs, files in os.walk(site_package):
        for file in files:
            if file.startswith('libclang.so'):
                libclang_path = os.path.join(root, file)
                break
        if libclang_path:
            break

print(f"Using libclang: {libclang_path}")
clang.cindex.Config.set_library_file(libclang_path)

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
        """Extract symbols from the library file using nm."""
        if not lib_path.exists():
            raise FileNotFoundError(f"Library file {lib_path} does not exist")

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
            self.intercept_str += f"DEFINE_EXPORT_C_REDIRECT_CALL(Driver::{function_name[self.prefix_len:]}, {return_type}, {function_name});\n"
            return

        self.driver_str += f"    DEFINE_STATIC_ADDRESS_CALL(GetSymbol(\"{function_name}\"), {return_type}, {function_name[self.prefix_len:]}, {', '.join(param_types)});\n"
        self.intercept_str += f"DEFINE_EXPORT_C_REDIRECT_CALL(Driver::{function_name[self.prefix_len:]}, {return_type}, {function_name}, {', '.join(param_types)});\n"

    def find_functions(self, node: clang.cindex.Cursor) -> None:
        """Recursively find and process function declarations."""
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            # Skip inline functions
            if node.is_definition() and "inline" in [token.spelling for token in node.get_tokens()]:
                print(f"Skipping inline function: {node.spelling}")
                return

            function_name = node.spelling
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

        with open(driver_file, "w") as f:
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

        with open(intercept_file, "w") as f:
            f.write(f"""/// This file is auto-generated by command \"{cmd}\"
#include "xsched/utils/function.h"
#include "xsched/{self.platform}/hal/driver.h"

using namespace xsched::{self.platform};

{self.intercept_str}
static const std::unordered_map<std::string, void *> intercept_symbol_map = {{
{self.intercept_entry_str}}};

DEFINE_DLSYM_INTERCEPT(intercept_symbol_map);
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate driver and intercept files for a given header file.')
    parser.add_argument('-s', '--source', type=str, required=True, help='Path to the header source')
    parser.add_argument('-I', '--include', type=str, action='append', help='Path to the include directory')
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
