# compare_policies_fixed.py - Fixed baseline comparison
import numpy as np
import matplotlib.pyplot as plt
from sagin_env import SAGINEnv
from sagin_env import SystemDownException


def run_policy(env, episodes=5):
    """Run baseline policy with corrected method names"""
    logs = {
        'success_log': [], 'energy': [], 'cache_hits': [], 'task_log': [], 'dropped': 0
    }

    try:
        for ep in range(episodes):
            print(f'Episode: {ep + 1}/{episodes}')

            for timestep in range(50):
                print(f"  Timestep: {timestep + 1}/50")

                try:
                    # Phase 1: IoT Data Collection
                    env.collect_iot_data()

                    # Phase 2: OFDM Slot Allocation (corrected method name)
                    env.allocate_ofdm_slots_with_constraints()

                    # Phase 3: Satellite Coverage Update
                    for sat in env.sats:
                        sat.update_coverage(timestep)

                    # Phase 4: UAV-to-Satellite Upload
                    env.upload_to_satellites()

                    # Phase 5: Satellite Synchronization
                    env.sync_satellites()

                    # Phase 6: Task Generation and Offloading
                    env.generate_and_offload_tasks()

                    # Phase 7: Execute one complete step (this includes task execution, cache updates, etc.)
                    # env.step()  # Comment this out since we're doing individual phases

                    # Instead, call the individual methods:
                    env.execute_all_tasks()
                    env.update_all_caches()
                    env.evict_expired_content()
                    env.monitor_system_health()

                    # Get performance metrics
                    performance = env.get_performance_summary()
                    logs['success_log'].append({
                        'completed': performance.get('total_tasks_completed', 0),
                        'successful': performance.get('total_tasks_successful', 0)
                    })

                except SystemDownException as e:
                    print(f"System down in episode {ep + 1}, timestep {timestep + 1}: {e}")
                    break
                except Exception as e:
                    print(f"Error in episode {ep + 1}, timestep {timestep + 1}: {e}")
                    break

    except SystemDownException as e:
        print(f"Simulation ended due to system failure: {e}")
    except Exception as e:
        print(f"Simulation ended due to error: {e}")

    # Collect final logs
    try:
        final_performance = env.get_performance_summary()
        logs['dropped'] = final_performance.get('dropped_tasks', 0)
        logs['success_log'] = env.success_log if hasattr(env, 'success_log') else []
        logs['task_log'] = env.task_log if hasattr(env, 'task_log') else []

        # Add some dummy energy and cache hit data if not available
        if not logs['energy']:
            logs['energy'] = [uav.energy for uav in env.uavs.values()]
        if not logs['cache_hits']:
            logs['cache_hits'] = [uav.cache_hits for uav in env.uavs.values()]

    except Exception as e:
        print(f"Warning: Could not collect final logs: {e}")

    return logs


def summarize_and_plot(baseline_log):
    """Generate summary and plots from baseline results"""
    try:
        def get_success_rate(log):
            if not log:
                return 0
            total = sum(e.get('completed', 0) for e in log if isinstance(e, dict))
            successful = sum(e.get('successful', 0) for e in log if isinstance(e, dict))
            return 100 * successful / total if total > 0 else 0

        # Calculate metrics
        success_rate = get_success_rate(baseline_log['success_log'])
        avg_cache_hit = np.mean(baseline_log['cache_hits']) if baseline_log['cache_hits'] else 0
        avg_energy_used = np.mean(baseline_log['energy']) if baseline_log['energy'] else 0
        dropped_tasks = baseline_log['dropped']

        print(f"\n📊 Baseline Results Summary:")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Average Cache Hits: {avg_cache_hit:.1f}")
        print(f"   Average Energy Used: {avg_energy_used:.0f}J")
        print(f"   Dropped Tasks: {dropped_tasks}")

        # Create visualization
        labels = ['Success Rate (%)', 'Avg Cache Hit', 'Avg Energy Used', 'Dropped Tasks']
        baseline_vals = [success_rate, avg_cache_hit, avg_energy_used, dropped_tasks]

        x = np.arange(len(labels))
        width = 0.4

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, baseline_vals, width, label='Baseline (Popularity-based Caching)',
                      color=['skyblue', 'lightgreen', 'orange', 'salmon'])

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Baseline Policy Performance", fontweight='bold')
        ax.legend()

        # Add value labels on bars
        for bar, value in zip(bars, baseline_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max(baseline_vals) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('baseline_policy_results.png', dpi=300, bbox_inches='tight')
        print(f"📈 Results saved as 'baseline_policy_results.png'")
        plt.show()

    except Exception as e:
        print(f"❌ Error in plotting: {e}")
        print("Basic results:")
        print(f"   Success log entries: {len(baseline_log.get('success_log', []))}")
        print(f"   Dropped tasks: {baseline_log.get('dropped', 0)}")


def main():
    """Main execution function"""
    print("🚀 SAGIN Baseline Policy Comparison")
    print("=" * 50)

    # Initialize environment parameters
    X, Y = 10, 10  # Grid size
    cache_size = 2048  # MB
    compute_power_uav = 30  # CPU cycles/second
    compute_power_sat = 200  # CPU cycles/second
    energy = 540000  # Joules
    max_queue = 10  # Maximum task queue length
    num_sats = 2  # Number of satellites
    num_iot_per_region = 50  # IoT devices per region
    max_active_iot = 25  # Maximum active IoT devices
    ofdm_slots = 9  # OFDM subchannels
    duration = 300  # Time slot duration (seconds)

    print(f"🔧 System Configuration:")
    print(f"   Grid: {X}x{Y} ({X * Y} UAVs)")
    print(f"   Cache Size: {cache_size}MB per UAV")
    print(f"   Energy: {energy:,}J per UAV")
    print(f"   Satellites: {num_sats}")
    print(f"   OFDM Slots: {ofdm_slots}")

    try:
        # Initialize SAGIN environment
        print(f"\n🏗️  Initializing SAGIN environment...")
        env = SAGINEnv(
            X=X, Y=Y, duration=duration, cache_size=cache_size,
            compute_power_uav=compute_power_uav, compute_power_sat=compute_power_sat,
            energy=energy, max_queue=max_queue, num_sats=num_sats,
            num_iot_per_region=num_iot_per_region, max_active_iot=max_active_iot,
            ofdm_slots=ofdm_slots
        )

        print(f"✅ Environment initialized successfully")
        print(f"   UAVs created: {len(env.uavs)}")
        print(f"   Satellites created: {len(env.sats)}")
        print(f"   IoT regions: {len(env.iot_regions)}")

        # Run baseline policy
        print(f"\n🎯 Running baseline policy simulation...")
        logs_baseline = run_policy(env, episodes=1)

        # Print detailed statistics
        print(f"\n=== Final UAV Task Statistics ===")
        if hasattr(env, 'task_stats') and 'uav' in env.task_stats:
            for coord, stats in env.task_stats['uav'].items():
                print(f"UAV {coord}: Generated={stats.get('generated', 0)}, "
                      f"Completed={stats.get('completed', 0)}, "
                      f"Successful={stats.get('successful', 0)}")
        else:
            print("No UAV task statistics available")

        print(f"\n=== Final Satellite Task Statistics ===")
        if hasattr(env, 'task_stats') and 'satellite' in env.task_stats:
            for sid, stats in env.task_stats['satellite'].items():
                print(f"Satellite {sid}: Received={stats.get('received', 0)}, "
                      f"Completed={stats.get('completed', 0)}, "
                      f"Successful={stats.get('successful', 0)}")
        else:
            print("No satellite task statistics available")

        # Generate summary and visualization
        print(f"\n📈 Generating results summary...")
        summarize_and_plot(logs_baseline)

        print(f"\n🎉 Baseline comparison completed successfully!")
        print(f"🔬 System is ready for RL agent comparison")

    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()