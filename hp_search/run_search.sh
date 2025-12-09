#!/bin/bash
# Hyperparameter search: mini
# Generated: 2025-08-15 17:14

source env/bin/activate

echo "🚀 Trial 1/2"
python src/main.py --config_path hp_search/config_01.yaml
echo
echo "🚀 Trial 2/2"
python src/main.py --config_path hp_search/config_02.yaml
echo
echo "✅ Search complete!"
