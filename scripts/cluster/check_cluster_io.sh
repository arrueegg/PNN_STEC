#!/bin/bash
# Debug script to check cluster storage performance

echo "==================================================="
echo "Cluster Storage Performance Diagnostic"
echo "==================================================="
echo ""

echo "1. Check available scratch locations:"
echo "   Local /scratch:"
df -h /scratch 2>/dev/null || echo "   ❌ /scratch not found"

echo ""
echo "   Local /tmp:"
df -h /tmp | head -3

echo ""
echo "   Cluster /cluster/work:"
df -h /cluster/work/igp_psr 2>/dev/null | head -3

echo ""
echo "2. Benchmark I/O speeds:"
echo "   Testing read speed from cluster storage..."

# Create temporary test file on cluster storage
TEST_FILE="/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/test_io.tmp"
TEST_SIZE="100M"

if command -v dd &> /dev/null; then
    echo "   Read speed (cluster /cluster/work):"
    dd if=/dev/zero bs=1M count=100 2>/dev/null | dd of=$TEST_FILE bs=1M 2>&1 | tail -1
    time dd if=$TEST_FILE of=/dev/null bs=1M 2>&1 | tail -1
    rm -f $TEST_FILE
else
    echo "   dd not available"
fi

echo ""
echo "3. Check where data is being cached:"
echo "   If local /scratch is being used, you should see copies there:"
ls -lh /scratch/*/data* 2>/dev/null || echo "   ❌ No data in /scratch (data is on network!)"

echo ""
echo "4. Recommendation:"
echo "   If you see data in /cluster/work but NOT in /scratch:"
echo "   - The move_to_scratch: true isn't working properly"
echo "   - Contact cluster admin for correct local scratch path"
echo "   - Update config with: scratch_dir: \"/scratch/\""

echo ""
echo "==================================================="
