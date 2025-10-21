#!/usr/bin/env python3
"""Build a small Madrigal HDF5 sample (first N dates/files) for quick inspection.

This script mirrors the production builder but stops after processing N Madrigal files/dates.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import h5py

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(Path(__file__).resolve().parents[1]))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.evaluation.madrigal_builder import build_sample



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--madrigal_path', required=True)
    p.add_argument('--out_h5', default='data/madrigal_sample.h5')
    p.add_argument('--split', choices=['train','val','test'], default='test')
    p.add_argument('--n_files', type=int, default=50)
    args = p.parse_args()
    out = build_sample(args.madrigal_path, args.out_h5, split=args.split, n_files=args.n_files)
    if out is None:
        print('No data found for given split/madrigal path; no file created')
    else:
        print(f'Created sample file: {out}')


if __name__ == '__main__':
    main()
