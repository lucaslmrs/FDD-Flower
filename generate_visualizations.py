#!/usr/bin/env python3
"""Script to generate all visualization plots after training."""

import sys
from pathlib import Path

# Add flower_app to path
sys.path.insert(0, str(Path(__file__).parent))

from flower_app.advanced_visualization import generate_all_visualizations

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate FL visualization plots")
    parser.add_argument("--run-dir", type=str, default=None,
                       help="Run directory (e.g., artifacts/run_005)")
    parser.add_argument("--results", type=str, default="results.json",
                       help="Path to results.json file")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save plots (defaults to run_dir)")
    
    args = parser.parse_args()
    
    # Determine paths
    if args.run_dir:
        run_dir = args.run_dir
        results_path = Path(run_dir) / "results.json" if (Path(run_dir) / "results.json").exists() else args.results
        save_dir = args.save_dir or run_dir
    else:
        # Find latest run
        artifacts = Path("artifacts")
        if artifacts.exists():
            runs = sorted([d for d in artifacts.iterdir() if d.is_dir() and d.name.startswith("run_")])
            if runs:
                run_dir = str(runs[-1])
                results_path = runs[-1] / "results.json"
                save_dir = args.save_dir or run_dir
                print(f"Using latest run: {run_dir}")
            else:
                run_dir = None
                results_path = args.results
                save_dir = args.save_dir or "artifacts"
        else:
            run_dir = None
            results_path = args.results
            save_dir = args.save_dir or "artifacts"
    
    # Generate visualizations
    generate_all_visualizations(str(results_path), run_dir, save_dir)
