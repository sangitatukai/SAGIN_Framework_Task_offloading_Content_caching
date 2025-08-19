# integrated_baseline_system.py - Complete Baseline System Integration
import numpy as np
import time
from sagin_env import SAGINEnv, SystemDownException

# Import all baseline components
from iot_aggregation_baselines import create_aggregation_baseline
from caching_offloading_baselines import create_caching_baseline, create_offloading_baseline
from ofdm_allocation_baselines import create_ofdm_baseline


class IntegratedBaselineSystem:
    """
    Integrated system that combines all four baseline approaches for complete comparison
    """

    def __init__(self, aggregation_type='greedy', caching_type='greedy',
                 offloading_type='content_aware', ofdm_type='priority_based'):
        """
        Initialize integrated baseline system

        Args:
            aggregation_type: 'greedy', 'gru_bandit', 'random'
            caching_type: 'greedy', 'stateless_ppo'
            offloading_type: 'random_split', 'content_aware', 'load_balanced'
            ofdm_type: 'backlog_greedy', 'round_robin', 'random', 'priority_based', 'fairness_aware'
        """
        self.aggregation_type = aggregation_type
        self.caching_type = caching_type
        self.offloading_type = offloading_type
        self.ofdm_type = ofdm_type

        # Baseline components (will be initialized with environment parameters)
        self.aggregation_baselines = {}  # One per UAV
        self.caching_baselines = {}  # One per UAV
        self.offloading_baselines = {}  # One per UAV
        self.ofdm_baseline = None  # One global

        print(f"🔧 Integrated Baseline System Configuration:")
        print(f"   IoT Aggregation: {aggregation_type}")
        print(f"   Content Caching: {caching_type}")
        print(f"   Task Offloading: {offloading_type}")
        print(f"   OFDM Allocation: {ofdm_type}")

    def initialize_baselines(self, env):
        """Initialize all baseline components with environment parameters"""

        # Initialize per-UAV baselines
        for uav_coord, uav in env.uavs.items():
            # IoT Aggregation baselines
            if self.aggregation_type == 'gru_bandit':
                self.aggregation_baselines[uav_coord] = create_aggregation_baseline(
                    self.aggregation_type, obs_dim=16, hidden_dim=32, duration=env.duration
                )
            else:
                self.aggregation_baselines[uav_coord] = create_aggregation_baseline(
                    self.aggregation_type, duration=env.duration
                )

            # Caching baselines
            if self.caching_type == 'stateless_ppo':
                self.caching_baselines[uav_coord] = create_caching_baseline(
                    self.caching_type, item_feature_dim=4, hidden_dim=32
                )
            else:
                self.caching_baselines[uav_coord] = create_caching_baseline(
                    self.caching_type, cache_capacity_mb=uav.cache_capacity_mb
                )

            # Offloading baselines
            self.offloading_baselines[uav_coord] = create_offloading_baseline(
                self.offloading_type
            )

        # Initialize global OFDM baseline
        self.ofdm_baseline = create_ofdm_baseline(
            self.ofdm_type,
            total_ofdm_slots=env.ofdm_slots,
            num_satellites=len(env.sats)
        )

        print(f"✅ Initialized baselines for {len(env.uavs)} UAVs and {len(env.sats)} satellites")

    def execute_baseline_step(self, env):
        """Execute one complete step using all baseline approaches"""

        print(f"\n🔄 Executing Baseline Step {env.g_timestep + 1}")

        # Phase 1: IoT Data Aggregation with Baselines
        self._execute_iot_aggregation_baselines(env)

        # Phase 2: OFDM Slot Allocation with Baselines
        self._execute_ofdm_allocation_baseline(env)

        # Phase 3: Satellite Coverage Update (unchanged)
        for sat in env.sats:
            sat.update_coverage(env.g_timestep)

        # Phase 4: UAV-to-Satellite Upload (unchanged)
        env.upload_to_satellites()

        # Phase 5: Satellite Synchronization (unchanged)
        env.sync_satellites()

        # Phase 6: Task Generation and Baseline Offloading
        self._execute_task_generation_and_offloading(env)

        # Phase 7: Task Execution (unchanged)
        env.execute_all_tasks()

        # Phase 8: Baseline Cache Updates
        self._execute_caching_baselines(env)

        # Phase 9: Content Eviction (unchanged)
        env.evict_expired_content()

        # Phase 10: System Monitoring (unchanged)
        env.monitor_system_health()

    def _execute_iot_aggregation_baselines(self, env):
        """Execute IoT aggregation using selected baseline"""
        env.g_timestep += 1
        print(f"\n=== IoT Aggregation ({self.aggregation_type}) ===")

        for (x, y), uav in env.uavs.items():
            # Get active IoT devices from region
            active_device_ids = env.iot_regions[(x, y)].sample_active_devices()

            if not active_device_ids:
                uav.aggregated_content = {}
                continue

            # Generate content
            content_list = env.iot_regions[(x, y)].generate_content(
                active_device_ids, env.g_timestep, grid_coord=(x, y)
            )
            content_dict = {tuple(c['id']): c for c in content_list}

            # Get interfering regions
            interfering_regions = [coord for coord in env.uavs.keys() if coord != (x, y)]

            # Use baseline aggregation
            aggregator = self.aggregation_baselines[(x, y)]

            if self.aggregation_type == 'gru_bandit':
                # Create temporal context (simplified)
                temporal_context = np.random.randn(16)  # Mock temporal context
                selected_devices, aggregated = aggregator.select_devices(
                    active_device_ids, content_dict, uav.uav_pos, interfering_regions,
                    temporal_context=temporal_context
                )
            else:
                selected_devices, aggregated = aggregator.select_devices(
                    active_device_ids, content_dict, uav.uav_pos, interfering_regions
                )

            # Update UAV with aggregated content
            uav.aggregated_content = aggregated

            print(f"UAV ({x},{y}): {len(selected_devices)}/{len(active_device_ids)} devices, "
                  f"{len(aggregated)} items aggregated")

    def _execute_ofdm_allocation_baseline(self, env):
        """Execute OFDM slot allocation using selected baseline"""
        print(f"\n=== OFDM Allocation ({self.ofdm_type}) ===")

        # Prepare UAV states for allocation decision
        uav_states = {}
        for (x, y), uav in env.uavs.items():
            uav_states[(x, y)] = {
                'aggregated_content': uav.aggregated_content,
                'queue': uav.queue,
                'energy_ratio': uav.energy / uav.max_energy
            }

        # Use baseline OFDM allocation
        slot_allocation, connected_uavs = self.ofdm_baseline.allocate_slots(uav_states)

        # Update environment state
        env.subchannel_assignments = slot_allocation
        env.connected_uavs = connected_uavs

    def _execute_task_generation_and_offloading(self, env):
        """Execute task generation and offloading using selected baseline"""
        print(f"\n=== Task Offloading ({self.offloading_type}) ===")

        for (x, y), uav in env.uavs.items():
            # Generate tasks (unchanged)
            tasks = uav.generate_tasks(env.X, env.Y, env.g_timestep)

            if (x, y) not in env.task_stats['uav']:
                env.task_stats['uav'][(x, y)] = {'generated': 0, 'completed': 0, 'successful': 0}

            # Process each task with baseline offloading
            offloader = self.offloading_baselines[(x, y)]

            for task in tasks:
                task['remaining_cpu'] = task['required_cpu']
                env.task_stats['uav'][(x, y)]['generated'] += 1
                uav.total_tasks += 1

                # Make baseline offloading decision
                if self.offloading_type == 'random_split':
                    # Simple random split
                    neighbor_coords = [coord for coord in env.uavs.keys()
                                       if coord != (x, y) and abs(coord[0] - x) + abs(coord[1] - y) == 1]
                    decisions = offloader.make_offloading_decisions(
                        [task], (x, y), neighbor_coords,
                        satellite_available=(x, y) in env.connected_uavs
                    )
                    offload_target = decisions.get(task['task_id'], 'local')

                else:
                    # Content-aware or load-balanced offloading
                    uav_state = {
                        'cache': set(uav.cache_storage.keys()),
                        'aggregated_content': set(uav.aggregated_content.keys()),
                        'queue': uav.queue,
                        'max_queue': uav.max_queue,
                        'energy_ratio': uav.energy / uav.max_energy
                    }

                    neighbor_states = {}
                    for coord in env.uavs.keys():
                        if coord != (x, y) and abs(coord[0] - x) + abs(coord[1] - y) == 1:
                            neighbor_uav = env.uavs[coord]
                            neighbor_states[coord] = {
                                'cache': set(neighbor_uav.cache_storage.keys()),
                                'aggregated_content': set(neighbor_uav.aggregated_content.keys()),
                                'queue': neighbor_uav.queue,
                                'max_queue': neighbor_uav.max_queue,
                                'energy_ratio': neighbor_uav.energy / neighbor_uav.max_energy
                            }

                    satellite_state = {
                        'available': (x, y) in env.connected_uavs,
                        'global_content_pool': set(env.global_satellite_content_pool.keys()),
                        'queue_length': sum(len(sat.task_queue) for sat in env.sats)
                    }

                    decisions = offloader.make_offloading_decisions(
                        [task], (x, y), uav_state, neighbor_states, satellite_state
                    )
                    offload_target = decisions.get(task['task_id'], 'local')

                # Execute offloading decision
                self._execute_offloading_decision(task, (x, y), offload_target, env)

                # Log decision
                env.task_log.append({
                    "timestep": env.g_timestep,
                    "task_id": task['task_id'],
                    "generated_by": (x, y),
                    "assigned_to": offload_target,
                    "content_id": task['content_id'],
                    "delay_bound": task['delay_bound']
                })

    def _execute_offloading_decision(self, task, uav_coord, target, env):
        """Execute the offloading decision"""
        cid = tuple(task['content_id'])
        task_id = task['task_id']

        if target == 'local':
            uav = env.uavs[uav_coord]
            success = uav.receive_task(task, from_coord=uav_coord)
            if success:
                print(f"[TASK {task_id}] Assigned to LOCAL UAV {uav_coord}")

        elif target.startswith('neighbor_'):
            # Parse neighbor coordinates
            parts = target.split('_')
            nx, ny = int(parts[1]), int(parts[2])
            if (nx, ny) in env.uavs:
                neighbor = env.uavs[(nx, ny)]
                success = neighbor.receive_task(task, from_coord=uav_coord)
                if success:
                    print(f"[TASK {task_id}] Offloaded to NEIGHBOR UAV ({nx},{ny})")

        elif target == 'satellite':
            # Find assigned satellite for this UAV
            uav_assignments = env.subchannel_assignments.get(uav_coord, {})
            for sat in env.sats:
                if uav_assignments.get(sat.sat_id, False):
                    sat.receive_task(task, from_coord=env.uavs[uav_coord].uav_pos)

                    if sat.sat_id not in env.task_stats['satellite']:
                        env.task_stats['satellite'][sat.sat_id] = {'received': 0, 'completed': 0, 'successful': 0}
                    env.task_stats['satellite'][sat.sat_id]['received'] += 1
                    print(f"[TASK {task_id}] Offloaded to SATELLITE {sat.sat_id}")
                    break

        else:
            # Task dropped
            env.dropped_tasks += 1
            print(f"[TASK {task_id}] DROPPED - {target}")

    def _execute_caching_baselines(self, env):
        """Execute caching updates using selected baseline"""
        print(f"\n=== Cache Updates ({self.caching_type}) ===")

        for (x, y), uav in env.uavs.items():
            # Evict expired content first (unchanged)
            uav.evict_expired_content(env.g_timestep)

            # Build candidate pool (same as paper Equation 22)
            candidate_pool = {}

            # Add existing cache content
            for cid, meta in uav.cache_storage.items():
                candidate_pool[cid] = meta

            # Add aggregated content
            for cid, content in uav.aggregated_content.items():
                candidate_pool[cid] = content

            # Add satellite content if connected
            is_connected = (x, y) in env.connected_uavs
            if is_connected:
                for cid, sat_meta in env.global_satellite_content_pool.items():
                    if cid not in candidate_pool:
                        meta = sat_meta.copy()
                        meta['timestamp'] = env.g_timestep
                        meta['from_satellite'] = True
                        candidate_pool[cid] = meta

            # Use baseline caching
            cacher = self.caching_baselines[(x, y)]

            if self.caching_type == 'stateless_ppo':
                selected_content = cacher.select_cache_content(candidate_pool, uav.cache_capacity_mb)
            else:
                selected_content = cacher.select_cache_content(candidate_pool)

            # Update UAV cache
            uav.cache_storage = selected_content
            uav.cache_used_mb = sum(content.get('size', 0) for content in selected_content.values())

            # Update popularity scores for greedy baseline
            if self.caching_type == 'greedy':
                for cid in selected_content:
                    if cid in uav.content_popularity:
                        cacher.update_popularity(cid, True)

            # Clear aggregated content
            uav.clear_aggregated_content()

            print(f"UAV ({x},{y}): Cached {len(selected_content)} items, "
                  f"{uav.cache_used_mb:.1f}/{uav.cache_capacity_mb}MB")


def run_baseline_comparison(baseline_configs, episodes=1, timesteps_per_episode=30):
    """
    Run comparison of different baseline configurations

    Args:
        baseline_configs: List of baseline configuration dictionaries
        episodes: Number of episodes to run
        timesteps_per_episode: Timesteps per episode

    Returns:
        comparison_results: Dictionary with results for each baseline
    """
    print("🔬 RUNNING BASELINE COMPARISON")
    print("=" * 80)

    comparison_results = {}

    for config_idx, config in enumerate(baseline_configs):
        config_name = f"{config['aggregation']}+{config['caching']}+{config['offloading']}+{config['ofdm']}"

        print(f"\n📊 Testing Configuration {config_idx + 1}/{len(baseline_configs)}: {config_name}")
        print("-" * 60)

        # Initialize environment (fresh for each configuration)
        env = SAGINEnv(
            X=3, Y=3, duration=300, cache_size=40, compute_power_uav=25,
            compute_power_sat=200, energy=80000, max_queue=15, num_sats=2,
            num_iot_per_region=20, max_active_iot=10, ofdm_slots=6
        )

        # Initialize baseline system
        baseline_system = IntegratedBaselineSystem(
            aggregation_type=config['aggregation'],
            caching_type=config['caching'],
            offloading_type=config['offloading'],
            ofdm_type=config['ofdm']
        )
        baseline_system.initialize_baselines(env)

        # Run simulation
        episode_results = []
        start_time = time.time()

        try:
            for episode in range(episodes):
                for timestep in range(timesteps_per_episode):
                    # Execute baseline step
                    baseline_system.execute_baseline_step(env)

                    # Collect performance
                    performance = env.get_performance_summary()
                    episode_results.append(performance)

                    # Check energy depletion
                    min_energy = min(uav.energy for uav in env.uavs.values())
                    if min_energy <= 0:
                        print(f"⚠️  Energy depleted at timestep {timestep + 1}")
                        break

        except SystemDownException as e:
            print(f"🛑 Simulation terminated: {e}")

        duration = time.time() - start_time

        # Calculate final metrics
        if episode_results:
            final_performance = episode_results[-1]
            avg_success_rate = np.mean([p['overall_success_rate'] for p in episode_results])
            avg_cache_hit_rate = np.mean([p['cache_hit_rate'] for p in episode_results])
            avg_energy_efficiency = np.mean([p['energy_efficiency'] for p in episode_results])

            comparison_results[config_name] = {
                'config': config,
                'final_success_rate': final_performance['overall_success_rate'],
                'avg_success_rate': avg_success_rate,
                'final_cache_hit_rate': final_performance['cache_hit_rate'],
                'avg_cache_hit_rate': avg_cache_hit_rate,
                'final_energy_efficiency': final_performance['energy_efficiency'],
                'avg_energy_efficiency': avg_energy_efficiency,
                'dropped_tasks': final_performance['dropped_tasks'],
                'timesteps_completed': len(episode_results),
                'duration': duration,
                'episode_results': episode_results
            }

            print(f"✅ Results: Success={avg_success_rate:.1f}%, "
                  f"Cache={avg_cache_hit_rate:.1f}%, "
                  f"Energy={avg_energy_efficiency:.4f}, "
                  f"Dropped={final_performance['dropped_tasks']}")
        else:
            comparison_results[config_name] = {
                'error': 'No results obtained',
                'duration': duration
            }

    return comparison_results


def print_comparison_summary(comparison_results):
    """Print summary of baseline comparison results"""
    print("\n" + "=" * 80)
    print("📊 BASELINE COMPARISON SUMMARY")
    print("=" * 80)

    if not comparison_results:
        print("❌ No results to display")
        return

    # Create comparison table
    headers = ["Configuration", "Success Rate", "Cache Hit Rate", "Energy Efficiency", "Dropped Tasks"]
    print(f"{headers[0]:<30} {headers[1]:<12} {headers[2]:<15} {headers[3]:<17} {headers[4]:<12}")
    print("-" * 90)

    # Sort by average success rate
    sorted_results = sorted(
        [(name, results) for name, results in comparison_results.items() if 'avg_success_rate' in results],
        key=lambda x: x[1]['avg_success_rate'], reverse=True
    )

    for config_name, results in sorted_results:
        success_rate = results['avg_success_rate']
        cache_hit_rate = results['avg_cache_hit_rate']
        energy_efficiency = results['avg_energy_efficiency']
        dropped_tasks = results['dropped_tasks']

        print(f"{config_name:<30} {success_rate:<12.1f} {cache_hit_rate:<15.1f} "
              f"{energy_efficiency:<17.4f} {dropped_tasks:<12}")

    # Find best performing configurations
    if sorted_results:
        best_config = sorted_results[0]
        print(f"\n🏆 Best Overall Configuration: {best_config[0]}")
        print(f"   Success Rate: {best_config[1]['avg_success_rate']:.1f}%")
        print(f"   Cache Hit Rate: {best_config[1]['avg_cache_hit_rate']:.1f}%")
        print(f"   Energy Efficiency: {best_config[1]['avg_energy_efficiency']:.4f}")


def main():
    """Main execution function for baseline comparison"""
    print("🚀 INTEGRATED BASELINE SYSTEM TESTING")
    print("=" * 80)

    # Define baseline configurations to test
    baseline_configs = [
        {
            'aggregation': 'greedy',
            'caching': 'greedy',
            'offloading': 'content_aware',
            'ofdm': 'priority_based'
        },
        {
            'aggregation': 'random',
            'caching': 'greedy',
            'offloading': 'random_split',
            'ofdm': 'backlog_greedy'
        },
        {
            'aggregation': 'greedy',
            'caching': 'greedy',
            'offloading': 'load_balanced',
            'ofdm': 'round_robin'
        },
        {
            'aggregation': 'greedy',
            'caching': 'greedy',
            'offloading': 'content_aware',
            'ofdm': 'random'
        }
    ]

    print(f"🔬 Testing {len(baseline_configs)} baseline configurations...")

    # Run comparison
    results = run_baseline_comparison(baseline_configs, episodes=1, timesteps_per_episode=20)

    # Print results
    print_comparison_summary(results)

    print(f"\n🎉 Baseline comparison complete!")
    print(f"📊 Results show performance differences between approaches")
    print(f"🚀 System ready for RL agent implementation and comparison!")

    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Baseline comparison interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during baseline comparison: {e}")
        import traceback

        traceback.print_exc()