# rl_demo_script.py - Quick demonstration of the RL-SAGIN system
import torch
import numpy as np
import matplotlib.pyplot as plt
from rl_sagin_integration import RLSAGINTrainer
from rl_formulation_sagin import HierarchicalSAGINAgent


def quick_rl_demo():
    """Quick demonstration of the RL system capabilities"""
    print("🤖 SAGIN Hierarchical RL System - Quick Demo")
    print("=" * 60)

    # Initialize the system with smaller parameters for quick demo
    print("🔧 Initializing system...")
    trainer = RLSAGINTrainer(
        grid_size=(2, 2),  # Smaller 2x2 grid for faster demo
        cache_size=30,  # Smaller cache
        num_episodes=50  # Fewer episodes for demo
    )

    print(f"✅ System initialized:")
    print(f"   Grid: 2x2 ({len(trainer.sagin_env.uavs)} UAVs)")
    print(f"   Satellites: {len(trainer.sagin_env.sats)}")
    print(f"   RL Parameters: {sum(p.numel() for p in trainer.rl_agent.parameters()):,}")

    # Quick training run
    print("\n🧠 Running quick training...")
    start_time = time.time()

    training_history = trainer.train(save_interval=50, plot_results=False)

    training_time = time.time() - start_time

    # Show training results
    if training_history['episode_rewards']:
        final_rewards = training_history['episode_rewards'][-10:]
        avg_reward = np.mean(final_rewards)

        print(f"\n📈 Training Results:")
        print(f"   Episodes completed: {len(training_history['episode_rewards'])}")
        print(f"   Final average reward: {avg_reward:.2f}")
        print(f"   Training time: {training_time:.1f}s")

    # Quick evaluation
    print("\n📊 Running evaluation...")
    eval_results = trainer.evaluate(num_episodes=5)

    print(f"✅ Evaluation Results:")
    print(f"   Success Rate: {eval_results['avg_success_rate']:.1f}%")
    print(f"   Cache Hit Rate: {eval_results['avg_cache_hit_rate']:.1f}%")
    print(f"   Energy Efficiency: {eval_results['avg_energy_efficiency']:.4f}")
    print(f"   Dropped Tasks: {eval_results['avg_dropped_tasks']:.1f}")

    # Plot simple results
    if training_history['episode_rewards']:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(training_history['episode_rewards'])
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        metrics = ['Success Rate', 'Cache Hit', 'Energy Eff.', 'Dropped Tasks']
        values = [eval_results['avg_success_rate'],
                  eval_results['avg_cache_hit_rate'],
                  eval_results['avg_energy_efficiency'] * 1000,  # Scale for visibility
                  eval_results['avg_dropped_tasks']]

        plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'salmon'])
        plt.title('Final Performance')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('rl_demo_results.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Demo results saved as 'rl_demo_results.png'")
        plt.show()

    print("\n🎉 Demo completed successfully!")
    return training_history, eval_results


def component_demonstration():
    """Demonstrate individual RL components"""
    print("\n🔬 Component-Level Demonstration")
    print("=" * 50)

    # 1. GRU Temporal Encoder Demo
    print("1️⃣ GRU Temporal Encoder:")
    from rl_formulation_sagin import GRUTemporalEncoder

    encoder = GRUTemporalEncoder(input_dim=16, hidden_dim=32)

    # Create sample observation
    sample_obs = encoder.create_observation_vector(
        zipf_param=1.8,
        active_devices=[1, 3, 5, 7],
        content_generation={1: 5.2, 3: 3.1, 5: 8.7},
        cache_hit_rate=0.65,
        energy_ratio=0.82,
        queue_length_ratio=0.3
    )

    # Forward pass
    temporal_embedding = encoder(sample_obs.unsqueeze(0))
    print(f"   Input shape: {sample_obs.shape}")
    print(f"   Output shape: {temporal_embedding.shape}")
    print(f"   ✅ Temporal encoder working")

    # 2. Graph Neural Network Demo
    print("\n2️⃣ Graph Neural Network Encoder:")
    from rl_formulation_sagin import GraphNeuralNetworkEncoder

    gnn = GraphNeuralNetworkEncoder(node_feature_dim=128, output_dim=64)

    # Sample UAV positions
    uav_positions = {(0, 0): (50, 50, 100), (0, 1): (50, 150, 100),
                     (1, 0): (150, 50, 100), (1, 1): (150, 150, 100)}

    edge_index, node_mapping = gnn.build_uav_graph(uav_positions)

    print(f"   UAVs: {len(uav_positions)}")
    print(f"   Edges: {edge_index.shape[1]}")
    print(f"   ✅ GNN encoder working")

    # 3. IoT Aggregation Agent Demo
    print("\n3️⃣ IoT Aggregation Agent:")
    from rl_formulation_sagin import IoTAggregationAgent

    iot_agent = IoTAggregationAgent(temporal_dim=32, max_devices=10)

    # Sample device features
    device_features = [
        iot_agent.create_device_features(i, size=np.random.uniform(1, 10),
                                         ttl=1200, transmission_time=0.5)
        for i in range(5)
    ]

    action_probs, state_value = iot_agent(temporal_embedding, device_features)

    print(f"   Devices: {len(device_features)}")
    print(f"   Action probs shape: {action_probs.shape}")
    print(f"   State value: {state_value.item():.3f}")
    print(f"   ✅ IoT aggregation agent working")

    # 4. MAPPO Agent Demo
    print("\n4️⃣ MAPPO Caching/Offloading Agent:")
    from rl_formulation_sagin import MAPPOCachingOffloadingAgent

    mappo_agent = MAPPOCachingOffloadingAgent(temporal_dim=32, spatial_dim=64)

    # Sample task burst
    task_burst = [{'required_cpu': 5, 'delay_bound': 10.0, 'size': 2.0} for _ in range(3)]
    task_features = mappo_agent.create_task_burst_features(task_burst, 0.8, 0.3)

    # Sample spatial embedding
    spatial_embedding = torch.randn(1, 64)

    concentration_params = mappo_agent.forward_offloading(
        temporal_embedding, spatial_embedding, task_features.unsqueeze(0)
    )

    print(f"   Task burst size: {len(task_burst)}")
    print(f"   Concentration params shape: {concentration_params.shape}")
    print(f"   ✅ MAPPO agent working")

    # 5. Centralized OFDM Agent Demo
    print("\n5️⃣ Centralized OFDM Agent:")
    from rl_formulation_sagin import CentralizedOFDMAgent

    ofdm_agent = CentralizedOFDMAgent(max_uavs=4, max_ofdm_slots=3)

    # Sample global state
    global_state = torch.randn(1, 16)  # 4 UAVs × 4 features

    logits, state_value = ofdm_agent(global_state)
    actions = ofdm_agent.sample_actions(logits, num_slots=3)

    print(f"   Global state shape: {global_state.shape}")
    print(f"   Action shape: {actions.shape}")
    print(f"   Selected UAVs: {actions.sum().item()}")
    print(f"   ✅ OFDM agent working")

    print(f"\n✅ All components working correctly!")


def architecture_overview():
    """Print system architecture overview"""
    print("\n🏗️ SYSTEM ARCHITECTURE OVERVIEW")
    print("=" * 50)

    print("📊 Hierarchical RL Structure:")
    print("   ┌─ GRU Temporal Encoder (Shared)")
    print("   │  └─ Captures IoT activation patterns")
    print("   │")
    print("   ├─ IoT Aggregation Agents (Single-Agent PPO)")
    print("   │  ├─ One per UAV region")
    print("   │  └─ Selects active IoT devices with TDMA constraints")
    print("   │")
    print("   ├─ Caching & Offloading Agents (Multi-Agent PPO)")
    print("   │  ├─ GNN for neighbor coordination")
    print("   │  ├─ Content caching decisions")
    print("   │  └─ Task offloading decisions")
    print("   │")
    print("   └─ OFDM Allocation Agent (Centralized PPO)")
    print("      └─ Global subchannel assignment")

    print(f"\n🧠 Key RL Features:")
    print(f"   • Temporal Dependencies: GRU captures time-varying patterns")
    print(f"   • Spatial Coordination: GNN enables UAV cooperation")
    print(f"   • Hierarchical Decomposition: 4 specialized sub-problems")
    print(f"   • Constraint Handling: TDMA, OFDM, energy, cache capacity")
    print(f"   • Multi-Objective: Task success + energy efficiency")

    print(f"\n📈 Expected Improvements over Baselines:")
    print(f"   • Adaptive IoT aggregation based on learned patterns")
    print(f"   • Cooperative caching reduces redundancy")
    print(f"   • Content-aware task offloading improves success rate")
    print(f"   • Dynamic OFDM allocation responds to load imbalances")


def main():
    """Main demo execution"""
    print("🚀 Welcome to the SAGIN Hierarchical RL Demo!")
    print("This demonstrates the complete RL system for SAGIN optimization")

    try:
        print("\n" + "=" * 60)
        architecture_overview()

        print("\n" + "=" * 60)
        component_demonstration()

        print("\n" + "=" * 60)
        training_history, eval_results = quick_rl_demo()

        print(f"\n🎯 Demo Summary:")
        print(f"   ✅ All RL components functional")
        print(f"   ✅ Training pipeline working")
        print(f"   ✅ Evaluation metrics computed")
        print(f"   ✅ Ready for full-scale experiments!")

        print(f"\n📚 Next Steps:")
        print(f"   1. Run full comparison: python rl_baseline_comparison.py")
        print(f"   2. Train longer: Increase num_episodes in RLSAGINTrainer")
        print(f"   3. Hyperparameter tuning: Adjust learning rates, architectures")
        print(f"   4. Ablation studies: Test individual component contributions")

    except Exception as e:
        print(f"\n❌ Demo encountered error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import time

    main()