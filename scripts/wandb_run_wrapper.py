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
    use_ipp_features = None  # Track if we need to expand IPP features

    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[1:][i]

        if arg.startswith("--") and "=" in arg:
            # This is a W&B parameter like --model.hidden_dim=256
            key_part, value_part = arg.split("=", 1)
            key = key_part[2:]  # Remove the '--' prefix

            # Special handling for use_ipp_features - expand to all 4 IPP feature flags
            if key == "use_ipp_features":
                use_ipp_features = value_part.lower() in ['true', '1', 'yes']
            else:
                # Pass through as --key=value (config parser handles this format)
                override_args.append(f"--{key}={value_part}")
        else:
            # Keep other arguments as-is
            remaining_args.append(arg)

        i += 1

    # If use_ipp_features was specified, expand it to all 4 IPP feature controls
    if use_ipp_features is not None:
        ipp_features = [
            "feature_control.lat_ipp",
            "feature_control.lon_ipp",
            "feature_control.sm_lat_ipp",
            "feature_control.sm_lon_ipp"
        ]
        # Convert boolean to string that config parser expects
        ipp_value = "true" if use_ipp_features else "false"
        for feature in ipp_features:
            override_args.append(f"--{feature}={ipp_value}")

    # Build the command to run main.py
    main_py_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    cmd = [sys.executable, main_py_path] + remaining_args + override_args

    print("🔄 Running main.py with converted arguments:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # Debug: Show IPP feature overrides if present
    if use_ipp_features is not None:
        print(f"📍 IPP features set to: {ipp_value}")
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
