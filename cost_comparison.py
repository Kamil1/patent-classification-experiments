#!/usr/bin/env python3
"""Cost comparison tool for analyzing optimization runs."""

import argparse
import json
import os
import glob
from typing import List, Dict
from cost_tracker import CostMetrics, CostComparator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CostOptimizationAnalyzer:
    """Analyze and compare costs across multiple optimization runs."""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = results_dir
        
    def find_cost_files(self, pattern: str = "*costs*.json") -> List[str]:
        """Find all cost metric files matching pattern."""
        pattern_path = os.path.join(self.results_dir, pattern)
        return glob.glob(pattern_path)
    
    def load_all_cost_metrics(self, file_paths: List[str] = None) -> List[Dict]:
        """Load cost metrics from multiple files."""
        if file_paths is None:
            file_paths = self.find_cost_files()
        
        metrics_data = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['file_path'] = file_path
                    data['run_name'] = self._extract_run_name(file_path)
                    metrics_data.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return metrics_data
    
    def _extract_run_name(self, file_path: str) -> str:
        """Extract run name from file path."""
        basename = os.path.basename(file_path)
        # Remove extension and timestamp
        name = basename.replace('.json', '')
        parts = name.split('_')
        if len(parts) >= 4:  # patent_classification_costs_split_timestamp
            return f"{parts[2]}_{parts[1]}"  # split_costs
        return basename
    
    def create_comparison_dataframe(self, metrics_data: List[Dict]) -> pd.DataFrame:
        """Create a pandas DataFrame for easy analysis."""
        rows = []
        for data in metrics_data:
            row = {
                'run_name': data['run_name'],
                'total_cost': data['total_estimated_cost_usd'],
                'cost_per_sample': data['cost_per_sample_usd'],
                'samples_per_second': data['samples_per_second'],
                'total_runtime': data['total_runtime'],
                'accuracy': 0,  # Will be filled from results if available
                'model_name': data['model_name'],
                'batch_size': data['batch_size'],
                'max_length': data['max_length'],
                'quantization': data['quantization'],
                'total_tokens': data['total_input_tokens'] + data['total_output_tokens'],
                'avg_input_tokens': data['avg_input_tokens'],
                'avg_output_tokens': data['avg_output_tokens'],
                'peak_gpu_memory_mb': data['peak_gpu_memory_mb'],
                'compute_cost': data['estimated_compute_cost_usd'],
                'token_cost': data['estimated_token_cost_usd'],
                'timestamp': data['timestamp'],
                'file_path': data['file_path']
            }
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def plot_cost_vs_performance(self, df: pd.DataFrame, save_path: str = None):
        """Plot cost vs performance scatter plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cost per sample vs Speed
        scatter1 = ax1.scatter(df['samples_per_second'], df['cost_per_sample'], 
                             s=df['batch_size']*10, alpha=0.7, c=df.index)
        ax1.set_xlabel('Samples per Second')
        ax1.set_ylabel('Cost per Sample (USD)')
        ax1.set_title('Cost vs Speed (bubble size = batch size)')
        
        # Add labels for each point
        for i, row in df.iterrows():
            ax1.annotate(row['run_name'], (row['samples_per_second'], row['cost_per_sample']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Total cost breakdown
        ax2.bar(df['run_name'], df['compute_cost'], label='Compute Cost', alpha=0.7)
        ax2.bar(df['run_name'], df['token_cost'], bottom=df['compute_cost'], 
                label='Token Cost', alpha=0.7)
        ax2.set_ylabel('Cost (USD)')
        ax2.set_title('Cost Breakdown by Run')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # Memory usage vs performance
        ax3.scatter(df['peak_gpu_memory_mb'], df['samples_per_second'], 
                   s=df['batch_size']*10, alpha=0.7)
        ax3.set_xlabel('Peak GPU Memory (MB)')
        ax3.set_ylabel('Samples per Second')
        ax3.set_title('Memory vs Speed')
        
        # Token efficiency
        ax4.scatter(df['avg_input_tokens'], df['cost_per_sample'], alpha=0.7)
        ax4.set_xlabel('Average Input Tokens')
        ax4.set_ylabel('Cost per Sample (USD)')
        ax4.set_title('Token Length vs Cost')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost analysis plot saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def find_pareto_optimal(self, df: pd.DataFrame, cost_col: str = 'cost_per_sample', 
                           performance_col: str = 'samples_per_second') -> pd.DataFrame:
        """Find Pareto optimal configurations (best cost-performance trade-offs)."""
        df_sorted = df.sort_values(cost_col)
        pareto_optimal = []
        
        max_performance = -float('inf')
        for _, row in df_sorted.iterrows():
            if row[performance_col] > max_performance:
                pareto_optimal.append(row)
                max_performance = row[performance_col]
                
        return pd.DataFrame(pareto_optimal)
    
    def generate_optimization_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive optimization report."""
        report = {
            'summary': {
                'total_runs': len(df),
                'best_cost': df.loc[df['cost_per_sample'].idxmin()],
                'best_speed': df.loc[df['samples_per_second'].idxmax()],
                'most_efficient': df.loc[(df['cost_per_sample'] * df['samples_per_second']).idxmin()]
            },
            'statistics': {
                'cost_range': (df['cost_per_sample'].min(), df['cost_per_sample'].max()),
                'speed_range': (df['samples_per_second'].min(), df['samples_per_second'].max()),
                'avg_cost': df['cost_per_sample'].mean(),
                'avg_speed': df['samples_per_second'].mean()
            },
            'correlations': {
                'batch_size_vs_speed': df['batch_size'].corr(df['samples_per_second']),
                'tokens_vs_cost': df['avg_input_tokens'].corr(df['cost_per_sample']),
                'memory_vs_speed': df['peak_gpu_memory_mb'].corr(df['samples_per_second'])
            },
            'pareto_optimal': self.find_pareto_optimal(df)
        }
        
        return report
    
    def print_optimization_report(self, report: Dict):
        """Print formatted optimization report."""
        print("\n" + "="*80)
        print("📊 COST OPTIMIZATION ANALYSIS REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"  Lowest Cost: ${summary['best_cost']['cost_per_sample']:.6f}/sample ({summary['best_cost']['run_name']})")
        print(f"  Fastest: {summary['best_speed']['samples_per_second']:.2f} samples/sec ({summary['best_speed']['run_name']})")
        print(f"  Most Efficient: {summary['most_efficient']['run_name']}")
        
        stats = report['statistics']
        print(f"\n📈 STATISTICS:")
        print(f"  Total Runs Analyzed: {stats['avg_cost']}")
        print(f"  Cost Range: ${stats['cost_range'][0]:.6f} - ${stats['cost_range'][1]:.6f}/sample")
        print(f"  Speed Range: {stats['speed_range'][0]:.2f} - {stats['speed_range'][1]:.2f} samples/sec")
        print(f"  Average Cost: ${stats['avg_cost']:.6f}/sample")
        print(f"  Average Speed: {stats['avg_speed']:.2f} samples/sec")
        
        corr = report['correlations']
        print(f"\n🔗 CORRELATIONS:")
        print(f"  Batch Size vs Speed: {corr['batch_size_vs_speed']:.3f}")
        print(f"  Input Tokens vs Cost: {corr['tokens_vs_cost']:.3f}")
        print(f"  GPU Memory vs Speed: {corr['memory_vs_speed']:.3f}")
        
        pareto = report['pareto_optimal']
        print(f"\n⭐ PARETO OPTIMAL CONFIGURATIONS ({len(pareto)} runs):")
        print(f"{'Run Name':<20} {'Cost/Sample':<15} {'Speed':<12} {'Config':<30}")
        print("-" * 80)
        for _, row in pareto.iterrows():
            config = f"bs={row['batch_size']}, {row['quantization']}"
            print(f"{row['run_name']:<20} ${row['cost_per_sample']:<14.6f} {row['samples_per_second']:<11.2f} {config:<30}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare patent classification costs")
    parser.add_argument('--results_dir', default='./results',
                       help='Directory containing cost metric files')
    parser.add_argument('--pattern', default='*costs*.json',
                       help='File pattern to match cost files')
    parser.add_argument('--plot', action='store_true',
                       help='Generate cost analysis plots')
    parser.add_argument('--save_plot', type=str,
                       help='Path to save the analysis plot')
    
    args = parser.parse_args()
    
    analyzer = CostOptimizationAnalyzer(args.results_dir)
    
    # Load all cost metrics
    print(f"🔍 Searching for cost files in {args.results_dir} with pattern '{args.pattern}'")
    cost_files = analyzer.find_cost_files(args.pattern)
    
    if not cost_files:
        print(f"❌ No cost files found matching pattern '{args.pattern}' in {args.results_dir}")
        return
    
    print(f"📊 Found {len(cost_files)} cost metric files")
    for file_path in cost_files:
        print(f"  - {file_path}")
    
    # Load and analyze data
    metrics_data = analyzer.load_all_cost_metrics(cost_files)
    if not metrics_data:
        print("❌ No valid cost data found")
        return
    
    df = analyzer.create_comparison_dataframe(metrics_data)
    report = analyzer.generate_optimization_report(df)
    
    # Print report
    analyzer.print_optimization_report(report)
    
    # Generate plots if requested
    if args.plot or args.save_plot:
        save_path = args.save_plot if args.save_plot else None
        analyzer.plot_cost_vs_performance(df, save_path)

if __name__ == "__main__":
    main()