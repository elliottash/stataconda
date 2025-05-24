import os
import sys
import unittest
import subprocess

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# List of all test files
TEST_FILES = [
    'test_command_abbreviations.py',
    'test_comments.py',
    'test_bash_commands.py',
    'test_extensive_analysis.py',
    'test_options.py',
    'test_regression_commands.py',
    'test_stataconda.py',
    'test_command_history.py',
    'test_line_continuation.py',
    'test_reghdfe.py',
    'test_graph.py',
    'test_highlighter.py',
]

def run_unittest_file(filename):
    print(f"\n===== Running {filename} with unittest =====")
    result = subprocess.run([sys.executable, os.path.join(TEST_DIR, filename)], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"FAILED: {filename}")
    else:
        print(f"PASSED: {filename}")
    return result.returncode

def main():
    failures = 0
    for test_file in TEST_FILES:
        if not os.path.exists(os.path.join(TEST_DIR, test_file)):
            print(f"SKIPPING: {test_file} (not found)")
            continue
        rc = run_unittest_file(test_file)
        if rc != 0:
            failures += 1
    if failures == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{failures} test file(s) failed.")

if __name__ == "__main__":
    main() 