
#!/bin/bash
echo "=== Phase 2 Constellation Demo ==="
echo "1. Training Pilot Model (10k steps)..."
python train_constellation.py

echo "2. Generating Visualizations..."
python visualize_constellation.py

echo "3. Artifacts Generated:"
echo "   - constellation_heatmap.png"
echo "   - constellation_demand.png"
echo "Done."
