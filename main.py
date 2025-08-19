# main.py - INTEGRATED SYSTEM MODEL WITH BASELINE APPROACHES
import numpy as np
import matplotlib.pyplot as plt
from sagin_env import SAGINEnv, SystemDownException
import time


def print_header(title):
    """Print formatted header"""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}")


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'-' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}")


def run_baseline_simulation(episodes=1, timesteps_per_episode=50):
    """
    Run the integrated SAGIN system with baseline approaches

    Baseline Approaches:
    1. IoT Aggregation: Spatiotemporal Zipf-based selection
    2. Caching: Popularity-based with TTL eviction
    3. Task Offloading: Content-aware (Local → Neighbor → Satellite → Drop)
    4. OFDM Allocation: Priority-based (content size + queue urgency)
    """

    print_header("SAGIN INTEGRATED SYSTEM MODEL - BASELINE SIMULATION")

    # System Parameters
    X, Y = 3, 3  # 3x3 grid
    cache_size = 40  # MB per UAV
    compute_power_uav = 25  # CPU cycles/second
    compute_power_sat = 200  # CPU cycles/second
    energy = 100000  # Joules per UAV
    max_queue = 15  # Maximum task queue length
    num_sats = 2  # Number of satellites
    num_iot_per_region = 20  # IoT devices per region
    max_active_iot = 12  # Maximum active IoT devices per slot
    ofdm_slots = 6  # Available OFDM subchannels
    duration = 300  # Time slot duration (seconds)

    print(f"🏗️  System Configuration:")
    print(f"   Grid Size: {X}x{Y} ({X * Y} UAVs)")
    print(f"   Cache Size: {cache_size}MB per UAV")
    print(f"   UAV Compute: {compute_power_uav} cycles/s")
    print(f"   Satellite Compute: {compute_power_sat} cycles/s")
    print(f"   Initial Energy: {energy:,}J per UAV")
    print(f"   Queue Capacity: {max_queue} tasks per UAV")
    print(f"   Satellites: {num_sats}")
    print(f"   IoT Devices: {num_iot_per_region} per region")
    print(f"   OFDM Subchannels: {ofdm_slots}")
    print(f"   Time Slot Duration: {duration}s")

    # Initialize environment
    print(f"\n🚀 Initializing SAGIN Environment...")
    env = SAGINEnv(
        X=X, Y=Y, duration=duration, cache_size=cache_size,
        compute_power_uav=compute_power_uav, compute_power_sat=compute_power_sat,
        energy=energy, max_queue=max_queue, num_sats=num_sats,
        num_iot_per_region=num_iot_per_region, max_active_iot=max_active_iot,
        ofdm_slots=ofdm_slots
    )

    # Storage for results
    episode_results = []

    for episode in range(episodes):
        print_section(f"EPISODE {episode + 1}/{episodes}")

        episode_performance = []
        episode_start_time = time.time()

        try:
            for timestep in range(timesteps_per_episode):
                step_start_time = time.time()

                print(f"\n🔄 Timestep {timestep + 1}/{timesteps_per_episode}")

                # Execute one complete timestep with all baseline approaches
                env.step()

                # Collect performance metrics
                performance = env.get_performance_summary()
                episode_performance.append(performance)

                step_duration = time.time() - step_start_time

                # Print timestep summary
                print(f"   ✅ Success Rate: {performance['overall_success_rate']:.1f}%")
                print(f"   📊 Cache Hit Rate: {performance['cache_hit_rate']:.1f}%")
                print(f"   ⚡ Energy Efficiency: {performance['energy_efficiency']:.4f} tasks/J")
                print(f"   ❌ Dropped Tasks: {performance['dropped_tasks']}")
                print(f"   🌐 Global Content Pool: {performance['global_content_pool_size']} items")
                print(f"   ⏱️  Step Duration: {step_duration:.2f}s")

                # Energy status check
                min_energy = min(uav.energy for uav in env.uavs.values())
                avg_energy = np.mean([uav.energy for uav in env.uavs.values()])
                max_energy_uav = max(env.uavs.values(), key=lambda u: u.energy)
                min_energy_pct = (min_energy / energy) * 100

                print(f"   🔋 Energy Status: Min={min_energy_pct:.1f}%, Avg={avg_energy:.0f}J")

                if min_energy_pct < 10:
                    print(f"   ⚠️  WARNING: Low energy detected!")

                # Detailed UAV status every 10 timesteps
                if (timestep + 1) % 10 == 0:
                    print(f"\n   📋 Detailed UAV Status (Timestep {timestep + 1}):")
                    for coord, uav in env.uavs.items():
                        status = uav.get_status_summary()
                        energy_pct = (uav.energy / energy) * 100
                        print(f"      UAV{coord}: Energy={energy_pct:.0f}%, "
                              f"Cache={status['cache_usage']}, "
                              f"Queue={status['queue_length']}, "
                              f"Tasks={status['tasks_completed']}")

                # Early termination check
                if min_energy <= 0:
                    print(f"   🛑 UAV energy depleted at timestep {timestep + 1}")
                    break

        except SystemDownException as e:
            print(f"🛑 Episode {episode + 1} terminated: {e}")

        episode_duration = time.time() - episode_start_time
        episode_results.append({
            'episode': episode + 1,
            'performance_history': episode_performance,
            'final_performance': episode_performance[-1] if episode_performance else None,
            'duration': episode_duration,
            'completed_timesteps': len(episode_performance)
        })

        print(f"\n📊 Episode {episode + 1} Summary:")
        if episode_performance:
            final_perf = episode_performance[-1]
            print(f"   Completed Timesteps: {len(episode_performance)}/{timesteps_per_episode}")
            print(f"   Final Success Rate: {final_perf['overall_success_rate']:.2f}%")
            print(f"   Final Cache Hit Rate: {final_perf['cache_hit_rate']:.2f}%")
            print(f"   Final Energy Efficiency: {final_perf['energy_efficiency']:.4f} tasks/J")
            print(f"   Total Dropped Tasks: {final_perf['dropped_tasks']}")
            print(f"   Episode Duration: {episode_duration:.1f}s")

    # Print comprehensive final summary
    env.print_final_summary()

    return env, episode_results


def plot_performance_analysis(episode_results):
    """Generate comprehensive performance plots"""

    print_header("PERFORMANCE ANALYSIS & VISUALIZATION")

    if not episode_results or not episode_results[0]['performance_history']:
        print("❌ No performance data to plot")
        return

    # Extract data for plotting
    performance_data = episode_results[0]['performance_history']  # Use first episode
    timesteps = [p['timestep'] for p in performance_data]
    success_rates = [p['overall_success_rate'] for p in performance_data]
    cache_hit_rates = [p['cache_hit_rate'] for p in performance_data]
    energy_efficiency = [p['energy_efficiency'] for p in performance_data]
    dropped_tasks = [p['dropped_tasks'] for p in performance_data]

    # Create comprehensive plot
    fig = plt.figure(figsize=(15, 10))

    # Success Rate Plot
    plt.subplot(2, 3, 1)
    plt.plot(timesteps, success_rates, 'b-o', linewidth=2, markersize=4)
    plt.title('Task Success Rate Over Time', fontweight='bold')
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Timestep')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    # Cache Hit Rate Plot
    plt.subplot(2, 3, 2)
    plt.plot(timesteps, cache_hit_rates, 'g-s', linewidth=2, markersize=4)
    plt.title('Cache Hit Rate Over Time', fontweight='bold')
    plt.ylabel('Cache Hit Rate (%)')
    plt.xlabel('Timestep')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    # Energy Efficiency Plot
    plt.subplot(2, 3, 3)
    plt.plot(timesteps, energy_efficiency, 'r-^', linewidth=2, markersize=4)
    plt.title('Energy Efficiency Over Time', fontweight='bold')
    plt.ylabel('Tasks per Joule')
    plt.xlabel('Timestep')
    plt.grid(True, alpha=0.3)

    # Dropped Tasks Plot
    plt.subplot(2, 3, 4)
    plt.plot(timesteps, dropped_tasks, 'orange', linewidth=2, marker='d', markersize=4)
    plt.title('Cumulative Dropped Tasks', fontweight='bold')
    plt.ylabel('Dropped Tasks')
    plt.xlabel('Timestep')
    plt.grid(True, alpha=0.3)

    # Performance Distribution (Box plot)
    plt.subplot(2, 3, 5)
    metrics_data = [success_rates, cache_hit_rates]
    plt.boxplot(metrics_data, labels=['Success Rate', 'Cache Hit Rate'])
    plt.title('Performance Distribution', fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)

    # System Metrics Summary
    plt.subplot(2, 3, 6)
    final_performance = performance_data[-1]
    metrics = ['Success\nRate (%)', 'Cache Hit\nRate (%)', 'Dropped\nTasks']
    values = [final_performance['overall_success_rate'],
              final_performance['cache_hit_rate'],
              final_performance['dropped_tasks']]

    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Final System Metrics', fontweight='bold')
    plt.ylabel('Value')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                 f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('sagin_baseline_performance.png', dpi=300, bbox_inches='tight')
    print(f"📈 Performance plots saved as 'sagin_baseline_performance.png'")
    plt.show()


def print_baseline_approaches_summary():
    """Print summary of baseline approaches used"""

    print_header("BASELINE APPROACHES IMPLEMENTED")

    print(f"🔬 The system implements the following baseline approaches:")
    print(f"")
    print(f"1️⃣  IoT Data Aggregation:")
    print(f"   • Spatiotemporal Zipf-based device activation (Equations 1-3)")
    print(f"   • TDMA protocol with interference-aware rate calculation")
    print(f"   • Content generation following paper specifications (Equation 4)")
    print(f"")
    print(f"2️⃣  Content Caching:")
    print(f"   • Popularity-based cache selection")
    print(f"   • TTL-based content eviction")
    print(f"   • Conditional candidate pool (Equation 22)")
    print(f"   • Capacity constraint enforcement (Equation 29)")
    print(f"")
    print(f"3️⃣  Task Offloading:")
    print(f"   • Content-aware offloading decisions")
    print(f"   • Priority order: Local → Neighbor → Satellite → Drop")
    print(f"   • Energy and queue-aware neighbor selection")
    print(f"   • Delay bound constraint checking (Equation 28)")
    print(f"")
    print(f"4️⃣  OFDM Slot Allocation:")
    print(f"   • Priority-based allocation (content size + queue urgency)")
    print(f"   • Constraint enforcement (Equation 23)")
    print(f"   • Load balancing across satellites")
    print(f"")
    print(f"5️⃣  Communication Protocols:")
    print(f"   • C-band IoT-UAV with TDMA (Equation 5)")
    print(f"   • Ka-band UAV-Satellite with OFDMA (Equations 8-9)")
    print(f"   • Proper energy consumption models (Equations 19-20)")
    print(f"")
    print(f"6️⃣  System Monitoring:")
    print(f"   • Energy depletion detection")
    print(f"   • Performance metrics tracking")
    print(f"   • System health monitoring")


def main():
    """Main execution function"""

    # Print system information
    print_baseline_approaches_summary()

    # Get user input for simulation parameters
    print_section("SIMULATION CONFIGURATION")

    try:
        episodes = int(input("Enter number of episodes (default: 1): ") or "1")
        timesteps = int(input("Enter timesteps per episode (default: 30): ") or "30")
    except ValueError:
        episodes = 1
        timesteps = 30
        print("Using default values: 1 episode, 30 timesteps")

    print(f"\n🎯 Configuration: {episodes} episode(s), {timesteps} timesteps each")

    # Run simulation
    start_time = time.time()
    env, episode_results = run_baseline_simulation(episodes=episodes, timesteps_per_episode=timesteps)
    total_duration = time.time() - start_time

    # Generate performance analysis
    plot_performance_analysis(episode_results)

    # Final system summary
    print_header("SIMULATION COMPLETE")

    print(f"🏁 Simulation Results:")
    print(f"   Total Simulation Time: {total_duration:.1f}s")
    print(f"   Episodes Completed: {len(episode_results)}")

    if episode_results:
        avg_timesteps = np.mean([r['completed_timesteps'] for r in episode_results])
        print(f"   Average Timesteps per Episode: {avg_timesteps:.1f}")

        if episode_results[0]['final_performance']:
            final_perf = episode_results[0]['final_performance']
            print(f"   Final Success Rate: {final_perf['overall_success_rate']:.2f}%")
            print(f"   Final Cache Hit Rate: {final_perf['cache_hit_rate']:.2f}%")
            print(f"   Final Energy Efficiency: {final_perf['energy_efficiency']:.4f} tasks/J")

    print(f"\n🎉 Baseline system model validation complete!")
    print(f"📊 Performance data saved and visualized")
    print(f"🚀 System is ready for RL agent implementation and comparison!")

    return env, episode_results


if __name__ == "__main__":
    try:
        env, results = main()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during simulation: {e}")
        import traceback

        traceback.print_exc()