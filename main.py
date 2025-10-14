#!/usr/bin/env python3
"""Main script for patent classification using Llama 8B."""

import argparse
import sys
import os
from config import Config
from pipeline import PatentClassificationPipeline
from pipeline_flexible import FlexiblePatentClassificationPipeline
from evaluate import PatentClassificationEvaluator
from data_loader import PatentDataLoader
from cost_comparison import CostOptimizationAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Patent Classification using Llama 8B")
    parser.add_argument('--mode', choices=['data', 'classify', 'evaluate', 'compare'], required=True,
                       help='Mode to run: data (explore dataset), classify (run inference), evaluate (analyze results), compare (compare costs)')
    parser.add_argument('--split', default='test', choices=['train', 'validation', 'test'],
                       help='Dataset split to use (default: test)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (default: all)')
    parser.add_argument('--results_path', type=str,
                       help='Path to results JSON file (required for evaluate mode)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference (default: 4)')
    parser.add_argument('--disable_cost_tracking', action='store_true',
                       help='Disable cost tracking and profiling')
    parser.add_argument('--compare_costs', action='store_true',
                       help='Compare costs from previous runs')
    parser.add_argument('--note', type=str, default=None,
                       help='Optional note to include in results and output')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name to use for classification (e.g., meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--model_type', type=str, choices=['auto', 'generative', 'classification'], 
                       default='auto', help='Model type: auto (detect), generative (Llama/GPT), or classification (BERT/RoBERTa)')
    
    args = parser.parse_args()
    
    # Create config with CLI overrides
    config = Config()
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.model:
        config.MODEL_NAME = args.model
    
    if args.mode == 'data':
        print("📊 Exploring patent classification dataset...")
        data_loader = PatentDataLoader(config)
        
        # Load and analyze dataset
        data_loader.load_dataset()
        data_loader.analyze_class_distribution()
        
        # Show sample data
        print("\n🔍 Sample data from training set:")
        samples = data_loader.get_sample_data('train', n_samples=3)
        for i, sample in enumerate(samples):
            print(f"\nExample {i+1}:")
            print(f"  Label: {sample['label']} - {sample['class_name']}")
            print(f"  Text: {sample['text']}")
            
    elif args.mode == 'classify':
        print("🤖 Running patent classification on Modal...")
        print(f"Model: {config.MODEL_NAME}")
        print(f"Model type: {args.model_type}")
        print(f"Split: {args.split}")
        print(f"Max samples: {args.max_samples or 'all'}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Cost tracking: {'disabled' if args.disable_cost_tracking else 'enabled'}")
        if args.note:
            print(f"📝 Note: {args.note}")
        print("🚀 Using Modal GPU acceleration (A10G)")
        
        try:
            # Use flexible pipeline that supports different model types
            pipeline = FlexiblePatentClassificationPipeline(
                config, 
                enable_cost_tracking=not args.disable_cost_tracking,
                model_type=args.model_type
            )
        except RuntimeError as e:
            print(f"\n❌ {e}")
            print("\n📋 To fix this issue:")
            print("   1. Install Modal: pip install modal")
            print("   2. Set up Modal auth: modal token new")
            print("   3. Deploy the app: modal deploy modal_inference.py")
            return
        
        # Run inference
        results = pipeline.run_inference(
            split=args.split,
            max_samples=args.max_samples,
            save_results=True,
            note=args.note
        )
        
        # Print analysis
        pipeline.print_analysis(args.split)
        
        print(f"\n✅ Classification complete! Results saved to {config.OUTPUT_DIR}/")
        
    elif args.mode == 'evaluate':
        if not args.results_path:
            print("❌ Error: --results_path is required for evaluate mode")
            sys.exit(1)
            
        if not os.path.exists(args.results_path):
            print(f"❌ Error: Results file not found: {args.results_path}")
            sys.exit(1)
            
        print(f"📈 Evaluating results from: {args.results_path}")
        
        evaluator = PatentClassificationEvaluator(config)
        metrics = evaluator.generate_report(args.results_path)
        
        print("\n✅ Evaluation complete! Check the output directory for plots and detailed metrics.")
    
    elif args.mode == 'compare':
        print("📊 Comparing costs from previous runs...")
        
        analyzer = CostOptimizationAnalyzer(config.OUTPUT_DIR)
        
        # Find cost files
        cost_files = analyzer.find_cost_files()
        if not cost_files:
            print(f"❌ No cost files found in {config.OUTPUT_DIR}")
            print("Run some classification tasks first with cost tracking enabled.")
            return
        
        print(f"Found {len(cost_files)} cost metric files:")
        for file_path in cost_files[-5:]:  # Show last 5
            print(f"  - {os.path.basename(file_path)}")
        if len(cost_files) > 5:
            print(f"  ... and {len(cost_files) - 5} more")
        
        # Load and analyze
        metrics_data = analyzer.load_all_cost_metrics(cost_files)
        if not metrics_data:
            print("❌ No valid cost data found")
            return
        
        df = analyzer.create_comparison_dataframe(metrics_data)
        report = analyzer.generate_optimization_report(df)
        
        # Print report
        analyzer.print_optimization_report(report)
        
        # Generate plots
        plot_path = os.path.join(config.OUTPUT_DIR, "cost_optimization_analysis.png")
        analyzer.plot_cost_vs_performance(df, plot_path)
        
        print(f"\n✅ Cost comparison complete! Analysis plot saved to {plot_path}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()