import re
import os
import argparse

def find_not_handled(class_name, file_path) -> dict:
    functions = {}

    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return []
    
    # Pattern to match: redirect xxx() -> Driver::xxx()
    # This will work with lines like:
    # [DEBG @ T2662363 @ 20:56:04.007208] redirect aclrtMemcpy() -> Driver::rtMemcpy() @ /home/ma-user/xproject/xsched/platforms/ascend/shim/src/intercept.cpp:99
    pattern = r'redirect\s+(\w+)\(\)\s*->\s*' + class_name + r'::(\w+)\(\)'

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                match = re.search(pattern, line)
                if match:
                    source_func = match.group(1)
                    functions[source_func] = functions.get(source_func, 0) + 1
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return functions

def main():
    parser = argparse.ArgumentParser(description="Find not handled functions.")
    parser.add_argument("file_path", help="Path to the log file.")
    parser.add_argument("-c", "--class_name", default="Driver", help="Class name to search for.")
    args = parser.parse_args()

    file_path = args.file_path
    matches = find_not_handled(args.class_name, file_path)
    if not matches:
        print("All functions are handled.")
        return
    
    print(f"Not handled functions (redirected to {args.class_name}) in {file_path}:")
    max_func_len = max(len(func) for func in matches.keys()) + 1
    for func, count in matches.items():
        print(f"{func:<{max_func_len}}: {count}")

if __name__ == "__main__":
    main()
