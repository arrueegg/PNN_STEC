#!/usr/bin/env python3
"""
W&B Sweep Wrapper Script
Converts W&B sweep parameters to the format expected by main.py
"""

import sys
import os
import subprocess
import argparse


def main():
    """
    Parse W&B command line arguments and convert them to --override format for main.py
    """
    argparse.ArgumentParser(description="W&B Sweep Wrapper", add_help=False)

    # Parse all arguments that match the pattern --key.subkey=value
    override_args = []
    remaining_args = []

    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1:][i]

        if arg.startswith("--") and "=" in arg:
            # This is a W&B parameter like --model.hidden_dim=256
            key_part, value_part = arg.split("=", 1)
            key = key_part[2:]  # Remove the '--' prefix

            # Convert to override format: --override key=value
            override_args.extend(["--override", f"{key}={value_part}"])
        else:
            # Keep other arguments as-is
            remaining_args.append(arg)

        i += 1

    # Build the command to run main.py
    main_py_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    cmd = [sys.executable, main_py_path] + remaining_args + override_args

    print("🔄 Running main.py with converted arguments:")
    print(f"   {' '.join(cmd)}")
    print()

    # Execute main.py with the converted arguments
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running main.py: {e}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
