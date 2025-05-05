#!/usr/bin/env python3

import re
from collections import defaultdict
import argparse
import subprocess
import os
import tempfile

def preprocess_file(file_path, include_paths):
    """Preprocess the file using clang with the specified include paths.
    
    Args:
        file_path: Path to the file to preprocess
        include_paths: List of include paths to pass to clang
        
    Returns:
        The preprocessed file content as a string
    """
    # Build the clang command with include paths
    cmd = ['c++', '-E', file_path]
    
    # Add each include path with -I prefix
    if include_paths:
        for path in include_paths:
            cmd.append(f'-I{path}')
    
    try:
        # Run clang preprocessor and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error preprocessing file: {e}")
        print(f"Stderr: {e.stderr}")
        raise

def extract_function_names(file_path):
    """Extract function names and versions from CUDA PFN type definitions."""
    
    # print(f"\nProcessing file: {file_path}")
    # print("\nFunction declarations found:")
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Regular expression to match PFN type definitions with version numbers
    # Matches patterns like: typedef CUresult ( *PFN_functionName_v####)
    pattern = r'typedef\s+CUresult\s+\( \*PFN_(\w+)_v(\d+)\)'
    
    # Store function names and their versions
    function_versions = defaultdict(set)
    
    # Find all matches
    matches = re.finditer(pattern, content)
    
    # Process matches
    for match in matches:
        function_name = match.group(1)
        version = match.group(2)
        function_versions[function_name].add(version)
    
    # Print results sorted by function name
    for func_name in sorted(function_versions.keys()):
        versions = sorted(function_versions[func_name])
        versions_str = ", ".join(f"{v}" for v in versions)
        # print(f"Function: {func_name} {versions_str}")
    
    # print(f"Total unique functions found: {len(function_versions)}")
    return function_versions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract function names and versions from CUDA PFN type definitions.')
    parser.add_argument('-f', type=str, required=True, help='Path to the CUDA header file')
    parser.add_argument('-I', type=str, action='append', help='Path to the include directory')
    args = parser.parse_args()

    # Preprocess the file first
    preprocessed_content = preprocess_file(args.f, args.I)
    
    # Write preprocessed content to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.h', delete=False) as tmp:
        tmp.write(preprocessed_content)
        tmp_path = tmp.name
    
    # Extract function names from the preprocessed file
    function_versions = extract_function_names(tmp_path)
    
    # Clean up temporary file

    os.unlink(tmp_path)

    """
    static std::unordered_map<std::string, std::map<int, void *>> intercept_funcs = {
    {"cuGetProcAddress"          , {{ 11030, (void *)cuGetProcAddress          }, { 12000, (void *)cuGetProcAddress_v2 }}},
    """

    max_func_name_len = max(len(func_name) for func_name in function_versions.keys())
    max_versions_len = [max_func_name_len]
    for func_name, versions in function_versions.items():
        for i, v in enumerate(versions):
            if i == 0:
                continue
            suffix = f"_v{i+1}"
            func_name_with_suffix = f"{func_name}{suffix}"
            func_len = len(func_name_with_suffix)
            if len(max_versions_len) <= i:
                max_versions_len.append(func_len)
                continue
            if func_len > max_versions_len[i]:
                max_versions_len[i] = func_len

    lines = []
    for func_name, versions in function_versions.items():
        sorted_versions = sorted(versions)
        version_str = []
        for i, v in enumerate(sorted_versions):
            if int(v) < 10000:
                v = " " + v
            suffix = f"_v{i+1}" if i > 0 else ""
            func_name_with_suffix = f"{func_name}{suffix}"
            version_str.append(f"{{ {v}, (void *){func_name_with_suffix:{max_versions_len[i]}}}}")
        quoted_func_name = f"\"{func_name}\""
        lines.append(f"{{ {quoted_func_name:{max_func_name_len+2}}, {{{', '.join(version_str)}}}}},")
    
    print("static const std::unordered_map<std::string, std::map<int, void *>> intercept_funcs = {")
    for line in lines:
        print(f"    {line}")
    print("};")
