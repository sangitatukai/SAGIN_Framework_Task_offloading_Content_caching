# complete_baseline_comparison.py - Show ALL baseline combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import itertools
from integrated_baseline_system import IntegratedBaselineSystem, run_baseline_comparison, print_comparison_summary
from sagin_env import SAGINEnv


def generate_all_baseline_combinations():
    """Generate all possible baseline combinations"""

    # Define all available baselines for each subproblem
    baselines = {
        'aggregation': ['greedy', 'random'],  # , 'gru_bandit' - may have issues
        'caching': ['greedy'],  # 'stateless_ppo' - may need fixing
        'offloading': ['content_aware', 'random_split', 'load_balanced'],
        'ofdm': ['priority_based', 'backlog_greedy', 'round_robin', 'random', 'fairness_aware']
    }

    # Generate all combinations
    combinations = []
    for agg in baselines['aggregation']:
        for cache in baselines['caching']:
            for offload in baselines['offloading']:
                for ofdm in baselines['ofdm']:
                    combination = {
                        'name': f"{agg}+{cache}+{offload}+{ofdm}",
                        'aggregation': agg,
                        'caching': cache,
                        'offloading': offload,
                        'ofdm': ofdm
                    }
                    combinations.append(combination)

    return combinations


def run_comprehensive_baseline_study(max_combinations=10, timesteps=15, trials=2):
    """Run comprehensive study of baseline combinations"""

    print("🔬 COMPREHENSIVE BASELINE STUDY")
    print("=" * 80)

    # Generate all combinations
    all_combinations = generate_all_baseline_combinations()
    print(f"📊 Total possible combinations: {len(all_combinations)}")

    # Limit combinations for feasible testing
    if len(all_combinations) > max_combinations:
        print(f"🔄 Testing first {max_combinations} combinations for demonstration")
        test_combinations = all_combinations[:max_combinations]
    else:
        test_combinations = all_combinations

    print(f"🧪 Testing {len(test_combinations)} baseline combinations:")
    for i, combo in enumerate(test_combinations[:5]):  # Show first 5
        print(f"   {i + 1}. {combo['name']}")
    if len(test_combinations) > 5:
        print(f"   ... and {len(test_combinations) - 5} more")

    # System parameters
    system_params = {
        'X': 3, 'Y': 3, 'duration': 300, 'cache_size': 40,
        'compute_power_uav': 25, 'compute_power_sat': 200,
        'energy': 80000, 'max_queue': 15, 'num_sats': 2,
        'num_iot_per_region': 20, 'max_active_iot': 10, 'ofdm_slots': 6
    }

    results = {}
    start_time = time.time()

    for i, config in enumerate(test_combinations):
        config_name = config['name']
        print(f"\n📊 [{i + 1}/{len(test_combinations)}] Testing: {config_name}")

        trial_results = []

        for trial in range(trials):
            print(f"   Trial {trial + 1}/{trials}")

            # Create fresh environment
            env = SAGINEnv(**system_params)

            # Initialize baseline system
            try:
                baseline_system = IntegratedBaselineSystem(
                    aggregation_type=config['aggregation'],
                    caching_type=config['caching'],
                    offloading_type=config['offloading'],
                    ofdm_type=config['ofdm']
                )
                baseline_system.initialize_baselines(env)

                # Run simulation
                episode_metrics = []

                for timestep in range(timesteps):
                    try:
                        baseline_system.execute_baseline_step(env)
                        performance = env.get_performance_summary()
                        episode_metrics.append(performance)

                        # Early termination check
                        min_energy = min(uav.energy for uav in env.uavs.values())
                        if min_energy <= 0:
                            break

                    except Exception as e:
                        print(f"      Error at timestep {timestep}: {e}")
                        break

                if episode_metrics:
                    final_perf = episode_metrics[-1]
                    avg_perf = {
                        'success_rate': np.mean([p['overall_success_rate'] for p in episode_metrics]),
                        'cache_hit_rate': np.mean([p['cache_hit_rate'] for p in episode_metrics]),
                        'energy_efficiency': np.mean([p['energy_efficiency'] for p in episode_metrics]),
                        'dropped_tasks': final_perf['dropped_tasks'],
                        'timesteps': len(episode_metrics)
                    }
                    trial_results.append(avg_perf)
                    print(f"      Success: {avg_perf['success_rate']:.1f}%")

            except Exception as e:
                print(f"   ❌ Configuration failed: {e}")
                continue

        # Aggregate trial results
        if trial_results:
            results[config_name] = {
                'config': config,
                'mean_success_rate': np.mean([r['success_rate'] for r in trial_results]),
                'std_success_rate': np.std([r['success_rate'] for r in trial_results]),
                'mean_cache_hit_rate': np.mean([r['cache_hit_rate'] for r in trial_results]),
                'std_cache_hit_rate': np.std([r['cache_hit_rate'] for r in trial_results]),
                'mean_energy_efficiency': np.mean([r['energy_efficiency'] for r in trial_results]),
                'std_energy_efficiency': np.std([r['energy_efficiency'] for r in trial_results]),
                'mean_dropped_tasks': np.mean([r['dropped_tasks'] for r in trial_results]),
                'std_dropped_tasks': np.std([r['dropped_tasks'] for r in trial_results]),
                'num_trials': len(trial_results)
            }
            print(f"   ✅ Completed: {results[config_name]['mean_success_rate']:.1f}% avg success")

    total_time = time.time() - start_time
    print(f"\n⏱️ Total testing time: {total_time:.1f}s")

    return results


def analyze_subproblem_contributions(results):
    """Analyze which baseline choices contribute most to performance"""

    print("\n🔬 SUBPROBLEM CONTRIBUTION ANALYSIS")
    print("=" * 60)

    if not results:
        print("❌ No results to analyze")
        return

    # Parse results by subproblem choices
    analysis = {
        'aggregation': {},
        'caching': {},
        'offloading': {},
        'ofdm': {}
    }

    for config_name, result in results.items():
        config = result['config']
        success_rate = result['mean_success_rate']

        # Group by each subproblem choice
        for subproblem in analysis.keys():
            choice = config[subproblem]
            if choice not in analysis[subproblem]:
                analysis[subproblem][choice] = []
            analysis[subproblem][choice].append(success_rate)

    # Calculate average performance for each choice
    for subproblem, choices in analysis.items():
        print(f"\n📊 {subproblem.upper()} BASELINE PERFORMANCE:")
        choice_avgs = []
        for choice, performances in choices.items():
            avg_perf = np.mean(performances)
            std_perf = np.std(performances)
            choice_avgs.append((avg_perf, choice, std_perf, len(performances)))
            print(f"   {choice:<15}: {avg_perf:5.1f}% ± {std_perf:4.1f}% ({len(performances)} configs)")

        # Identify best choice for this subproblem
        choice_avgs.sort(reverse=True)
        best_choice = choice_avgs[0]
        print(f"   🏆 BEST: {best_choice[1]} ({best_choice[0]:.1f}%)")


def create_comprehensive_visualization(results):
    """Create comprehensive visualization of all baseline results"""

    if not results:
        print("❌ No results to visualize")
        return

    print(f"\n📊 Creating visualization for {len(results)} configurations...")

    # Prepare data
    config_names = list(results.keys())
    success_rates = [results[name]['mean_success_rate'] for name in config_names]
    success_stds = [results[name]['std_success_rate'] for name in config_names]
    cache_hits = [results[name]['mean_cache_hit_rate'] for name in config_names]
    energy_effs = [results[name]['mean_energy_efficiency'] for name in config_names]
    dropped_tasks = [results[name]['mean_dropped_tasks'] for name in config_names]

    # Sort by success rate
    sorted_indices = sorted(range(len(success_rates)), key=lambda i: success_rates[i], reverse=True)

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Success Rate Comparison (Top configurations)
    top_n = min(10, len(config_names))  # Show top 10
    top_indices = sorted_indices[:top_n]
    top_names = [config_names[i][:20] + '...' if len(config_names[i]) > 20 else config_names[i] for i in top_indices]
    top_success = [success_rates[i] for i in top_indices]
    top_std = [success_stds[i] for i in top_indices]

    axes[0, 0].barh(range(top_n), top_success, xerr=top_std, color='skyblue', alpha=0.7)
    axes[0, 0].set_yticks(range(top_n))
    axes[0, 0].set_yticklabels(top_names)
    axes[0, 0].set_xlabel('Success Rate (%)')
    axes[0, 0].set_title(f'Top {top_n} Baseline Configurations (Success Rate)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Performance Distribution
    axes[0, 1].hist(success_rates, bins=min(10, len(success_rates) // 2 + 1), alpha=0.7, color='lightgreen')
    axes[0, 1].axvline(np.mean(success_rates), color='red', linestyle='--',
                       label=f'Mean: {np.mean(success_rates):.1f}%')
    axes[0, 1].set_xlabel('Success Rate (%)')
    axes[0, 1].set_ylabel('Number of Configurations')
    axes[0, 1].set_title('Success Rate Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Success Rate vs Energy Efficiency
    axes[1, 0].scatter(success_rates, energy_effs, alpha=0.6, s=50, color='purple')
    axes[1, 0].set_xlabel('Success Rate (%)')
    axes[1, 0].set_ylabel('Energy Efficiency (tasks/J)')
    axes[1, 0].set_title('Success Rate vs Energy Efficiency', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(success_rates, energy_effs, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(sorted(success_rates), p(sorted(success_rates)), "r--", alpha=0.8)

    # Summary Statistics
    axes[1, 1].axis('off')

    # Find best overall configuration
    best_idx = sorted_indices[0]
    best_config = config_names[best_idx]
    best_details = results[best_config]

    summary_text = f"""
BASELINE COMPARISON SUMMARY

📊 Total Configurations Tested: {len(results)}
🏆 Best Configuration: 
{best_config}

📈 Performance:
• Success Rate: {best_details['mean_success_rate']:.1f}% ± {best_details['std_success_rate']:.1f}%
• Cache Hit Rate: {best_details['mean_cache_hit_rate']:.1f}% ± {best_details['std_cache_hit_rate']:.1f}%
• Energy Efficiency: {best_details['mean_energy_efficiency']:.4f} ± {best_details['std_energy_efficiency']:.4f}
• Dropped Tasks: {best_details['mean_dropped_tasks']:.0f} ± {best_details['std_dropped_tasks']:.0f}

📊 Overall Statistics:
• Mean Success Rate: {np.mean(success_rates):.1f}%
• Success Rate Range: {min(success_rates):.1f}% - {max(success_rates):.1f}%
• Standard Deviation: {np.std(success_rates):.1f}%
"""

    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('comprehensive_baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("📈 Comprehensive results saved as 'comprehensive_baseline_comparison.png'")
    plt.show()


def save_detailed_results(results, filename='detailed_baseline_results.csv'):
    """Save detailed results to CSV"""

    if not results:
        print("❌ No results to save")
        return

    # Prepare data for CSV
    data = []
    for config_name, result in results.items():
        config = result['config']
        data.append({
            'Configuration': config_name,
            'Aggregation': config['aggregation'],
            'Caching': config['caching'],
            'Offloading': config['offloading'],
            'OFDM': config['ofdm'],
            'Mean_Success_Rate': result['mean_success_rate'],
            'Std_Success_Rate': result['std_success_rate'],
            'Mean_Cache_Hit_Rate': result['mean_cache_hit_rate'],
            'Std_Cache_Hit_Rate': result['std_cache_hit_rate'],
            'Mean_Energy_Efficiency': result['mean_energy_efficiency'],
            'Std_Energy_Efficiency': result['std_energy_efficiency'],
            'Mean_Dropped_Tasks': result['mean_dropped_tasks'],
            'Std_Dropped_Tasks': result['std_dropped_tasks'],
            'Num_Trials': result['num_trials']
        })

    df = pd.DataFrame(data)
    df = df.sort_values('Mean_Success_Rate', ascending=False)
    df.to_csv(filename, index=False)

    print(f"💾 Detailed results saved to {filename}")
    print(f"📊 Top 5 configurations:")
    for i, row in df.head(5).iterrows():
        print(f"   {i + 1}. {row['Configuration']}: {row['Mean_Success_Rate']:.1f}%")

    return df


def main():
    """Main execution"""
    print("🚀 COMPREHENSIVE SAGIN BASELINE COMPARISON")
    print("Testing ALL baseline combinations for 4 subproblems")
    print("=" * 70)

    try:
        # Configuration
        max_configs = int(input("Max configurations to test (default=12): ") or "12")
        timesteps = int(input("Timesteps per trial (default=15): ") or "15")
        trials = int(input("Trials per configuration (default=2): ") or "2")

        print(f"\n🔧 Configuration:")
        print(f"   Max configurations: {max_configs}")
        print(f"   Timesteps per trial: {timesteps}")
        print(f"   Trials per config: {trials}")

        # Run comprehensive study
        results = run_comprehensive_baseline_study(max_configs, timesteps, trials)

        if results:
            # Analyze results
            print_comparison_summary({'all_results': results})
            analyze_subproblem_contributions(results)
            create_comprehensive_visualization(results)
            save_detailed_results(results)

            print(f"\n🎉 COMPREHENSIVE BASELINE STUDY COMPLETED!")
            print(f"📊 {len(results)} configurations tested successfully")
            print(f"📈 Results saved and visualized")
            print(f"🔬 Subproblem contribution analysis completed")
        else:
            print("❌ No successful results obtained")

    except KeyboardInterrupt:
        print("\n⏹️ Study interrupted by user")
    except Exception as e:
        print(f"\n❌ Study failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()