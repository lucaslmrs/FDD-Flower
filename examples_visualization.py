#!/usr/bin/env python3
"""
Example script demonstrating how to use the advanced visualization module.

This script shows different ways to generate visualizations from FL results.
"""

from pathlib import Path
from flower_app.advanced_visualization import (
    generate_all_visualizations,
    plot_federated_convergence,
    plot_client_accuracy_evolution,
    plot_global_vs_personalized,
    plot_client_sample_distribution,
    plot_model_parameters_analysis,
    plot_confusion_matrices,
    load_results_json
)


def example_1_generate_all():
    """Example 1: Generate all visualizations at once."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Generate All Visualizations")
    print("="*60)
    
    # Find latest run
    artifacts = Path("artifacts")
    if artifacts.exists():
        runs = sorted([d for d in artifacts.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if runs:
            latest_run = runs[-1]
            print(f"Using latest run: {latest_run}")
            
            results_path = latest_run / "results.json"
            generate_all_visualizations(
                results_path=str(results_path),
                run_dir=str(latest_run),
                save_dir=str(latest_run)
            )
        else:
            print("No runs found in artifacts/")
    else:
        print("artifacts/ directory not found")


def example_2_individual_plots():
    """Example 2: Generate individual plots."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Generate Individual Plots")
    print("="*60)
    
    # Load results
    results_path = "results.json"
    if not Path(results_path).exists():
        print(f"File {results_path} not found")
        return
    
    results = load_results_json(results_path)
    save_dir = "custom_plots"
    
    # Generate specific plots
    print("\nGenerating federated convergence plot...")
    plot_federated_convergence(results, save_dir)
    
    print("\nGenerating global vs personalized comparison...")
    plot_global_vs_personalized(results, save_dir)
    
    print("\nGenerating client sample distribution...")
    plot_client_sample_distribution(save_dir)
    
    print(f"\nPlots saved to: {save_dir}/")


def example_3_custom_run():
    """Example 3: Specify a custom run directory."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Run Directory")
    print("="*60)
    
    run_dir = "artifacts/run_005"  # Change this to your run
    
    if not Path(run_dir).exists():
        print(f"Directory {run_dir} not found")
        return
    
    results_path = Path(run_dir) / "results.json"
    
    if not results_path.exists():
        print(f"File {results_path} not found")
        return
    
    # Generate all visualizations for this specific run
    generate_all_visualizations(
        results_path=str(results_path),
        run_dir=run_dir,
        save_dir=run_dir
    )


def example_4_only_confusion_matrices():
    """Example 4: Generate only confusion matrices."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Only Confusion Matrices")
    print("="*60)
    
    run_dir = "artifacts/run_005"  # Change this
    
    if not Path(run_dir).exists():
        print(f"Directory {run_dir} not found")
        return
    
    print(f"\nGenerating confusion matrices for {run_dir}...")
    plot_confusion_matrices(run_dir, save_dir=run_dir)


def example_5_model_analysis():
    """Example 5: Analyze model structure."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Model Structure Analysis")
    print("="*60)
    
    print("\nAnalyzing frozen vs trainable parameters...")
    plot_model_parameters_analysis(save_dir="model_analysis")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        examples = {
            "1": example_1_generate_all,
            "2": example_2_individual_plots,
            "3": example_3_custom_run,
            "4": example_4_only_confusion_matrices,
            "5": example_5_model_analysis,
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print("Available examples: 1, 2, 3, 4, 5")
    else:
        # Show menu
        print("\n" + "="*60)
        print("VISUALIZATION EXAMPLES")
        print("="*60)
        print("\nAvailable examples:")
        print("  1. Generate all visualizations (latest run)")
        print("  2. Generate individual plots")
        print("  3. Specify custom run directory")
        print("  4. Generate only confusion matrices")
        print("  5. Analyze model structure")
        print("\nUsage: python examples_visualization.py <example_number>")
        print("Example: python examples_visualization.py 1")
        print("\nOr use the main script:")
        print("  python generate_visualizations.py [options]")
