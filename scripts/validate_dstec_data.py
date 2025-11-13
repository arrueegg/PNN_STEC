#!/usr/bin/env python3
"""
Quick validation script to check if dSTEC evaluation will work with your data.

This script verifies that your H5 test data contains all required fields for
dSTEC evaluation.
"""

import sys
import h5py
import argparse
from pathlib import Path

# Required fields for dSTEC evaluation
REQUIRED_FIELDS = [
    "lat_sta",     # Station latitude
    "lon_sta",     # Station longitude  
    "satele",      # Satellite elevation
    "sod",         # Seconds of day
    "gfphase",     # Geometry-free phase (ground truth)
    "slipc",       # Slip cycle indicator
    "stec",        # STEC target (will be replaced by prediction)
    "station",     # Station identifier (required for pass identification)
    "sat",         # Satellite ID (required for pass identification)
    "year",        # Year (required for temporal pass separation)
    "doy",         # Day of year (required for temporal pass separation)
]

OPTIONAL_FIELDS = [
    "satazi",      # Satellite azimuth (for future use)
]


def check_h5_file(h5_path: Path) -> bool:
    """
    Check if H5 file contains required fields.
    
    Returns:
        True if all required fields present, False otherwise
    """
    print(f"Checking: {h5_path}")
    print("-" * 80)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Find the data array (try common paths)
            data = None
            if 'data' in f:
                data = f['data']
                print(f"✓ Found dataset: /data")
            else:
                # Try looking for year/doy structure
                for key in f.keys():
                    if key.isdigit():  # year
                        for doy_key in f[key].keys():
                            if 'all_data' in f[key][doy_key]:
                                data = f[key][doy_key]['all_data']
                                print(f"✓ Found dataset: /{key}/{doy_key}/all_data")
                                break
                        if data is not None:
                            break
            
            if data is None:
                print("❌ Could not find data array in H5 file")
                print("   Expected paths: '/data' or '/<year>/<doy>/all_data'")
                return False
            
            # Get field names
            if hasattr(data, 'dtype') and data.dtype.names:
                fields = list(data.dtype.names)
                print(f"\n📋 Available fields ({len(fields)}):")
                for field in sorted(fields):
                    print(f"   - {field}")
            else:
                print("❌ Data is not a structured array (no named fields)")
                return False
            
            # Check required fields
            print(f"\n🔍 Checking required fields:")
            all_present = True
            for field in REQUIRED_FIELDS:
                if field in fields:
                    print(f"   ✓ {field}")
                else:
                    print(f"   ❌ {field} - MISSING!")
                    all_present = False
            
            # Check optional fields
            print(f"\n🔍 Checking optional fields:")
            for field in OPTIONAL_FIELDS:
                if field in fields:
                    print(f"   ✓ {field}")
                else:
                    print(f"   ⚠️  {field} - not present (optional)")
            
            print("")
            if all_present:
                print("✅ All required fields present - dSTEC evaluation will work!")
                return True
            else:
                print("❌ Missing required fields - dSTEC evaluation will fail!")
                print("\nTo fix: Ensure your data preprocessing includes all required fields.")
                print("Critical fields: station, sat, slipc, year, doy (for unique pass identification)")
                return False
                
    except Exception as e:
        print(f"❌ Error reading H5 file: {e}")
        return False


def find_test_h5_files(data_dir: Path) -> list:
    """Find test H5 files in data directory."""
    test_files = []
    
    # Look for files with "test" in name
    for pattern in ["*test*.h5", "*TEST*.h5", "test_*.h5"]:
        test_files.extend(data_dir.glob(pattern))
    
    return sorted(set(test_files))


def main():
    parser = argparse.ArgumentParser(
        description="Validate data files for dSTEC evaluation"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing H5 data files (default: data/)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Specific H5 file to check (overrides --data-dir)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("dSTEC Data Validation Tool")
    print("=" * 80)
    print("")
    
    if args.file:
        # Check specific file
        if not args.file.exists():
            print(f"❌ File not found: {args.file}")
            return 1
        
        success = check_h5_file(args.file)
        return 0 if success else 1
    
    else:
        # Check all test files in directory
        if not args.data_dir.exists():
            print(f"❌ Data directory not found: {args.data_dir}")
            return 1
        
        test_files = find_test_h5_files(args.data_dir)
        
        if not test_files:
            print(f"❌ No test H5 files found in {args.data_dir}")
            print("   Looking for files matching: *test*.h5, *TEST*.h5, test_*.h5")
            return 1
        
        print(f"Found {len(test_files)} test file(s):\n")
        
        results = []
        for h5_file in test_files:
            success = check_h5_file(h5_file)
            results.append(success)
            print("")
        
        # Summary
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        n_ok = sum(results)
        n_total = len(results)
        print(f"Files checked: {n_total}")
        print(f"Files OK: {n_ok}")
        print(f"Files with issues: {n_total - n_ok}")
        
        return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
