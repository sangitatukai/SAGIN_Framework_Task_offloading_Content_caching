# test_rl_components.py - Test RL components independently
import torch
import numpy as np
from rl_formulation_sagin import (
    GRUTemporalEncoder,
    GraphNeuralNetworkEncoder,
    IoTAggregationAgent,
    MAPPOCachingOffloadingAgent,
    CentralizedOFDMAgent,
    HierarchicalSAGINAgent,
    test_hierarchical_agent
)


def test_individual_components():
    """Test each component individually"""
    print("🔬 Testing Individual RL Components")
    print("=" * 50)

    # Test 1: GRU Temporal Encoder
    print("1️⃣ Testing GRU Temporal Encoder...")
    temporal_encoder = GRUTemporalEncoder(input_dim=16, hidden_dim=64)

    # Create sample observation
    obs = temporal_encoder.create_observation_vector(
        zipf_param=1.8,
        active_devices=[1, 3, 5, 7],
        content_generation={1: 5.2, 3: 3.1, 5: 8.7},
        cache_hit_rate=0.65,
        energy_ratio=0.82,
        queue_length_ratio=0.3
    )

    print(f"   Observation shape: {obs.shape}")

    # Test forward pass
    temporal_embedding = temporal_encoder(obs.unsqueeze(0))
    print(f"   Temporal embedding shape: {temporal_embedding.shape}")
    print("   ✅ GRU Temporal Encoder working")

    # Test 2: Graph Neural Network
    print("\n2️⃣ Testing Graph Neural Network...")
    gnn = GraphNeuralNetworkEncoder(node_feature_dim=128, output_dim=64)

    # Sample UAV positions
    uav_positions = {
        (0, 0): (50, 50, 100),
        (0, 1): (50, 150, 100),
        (1, 0): (150, 50, 100),
        (1, 1): (150, 150, 100)
    }

    edge_index, node_mapping = gnn.build_uav_graph(uav_positions)
    print(f"   UAVs: {len(uav_positions)}")
    print(f"   Edges: {edge_index.shape[1] if edge_index.numel() > 0 else 0}")
    print(f"   Node mapping: {node_mapping}")

    # Test GNN forward pass
    node_features = torch.randn(len(uav_positions), 128)
    spatial_embeddings = gnn(node_features, edge_index)
    print(f"   Spatial embeddings shape: {spatial_embeddings.shape}")
    print("   ✅ GNN working")

    # Test 3: IoT Aggregation Agent
    print("\n3️⃣ Testing IoT Aggregation Agent...")
    iot_agent = IoTAggregationAgent(temporal_dim=64, max_devices=10)

    # Sample device features
    device_features = [
        iot_agent.create_device_features(i, size=np.random.uniform(1, 10),
                                         ttl=1200, transmission_time=0.5)
        for i in range(5)
    ]

    action_probs, state_value = iot_agent(temporal_embedding, device_features)
    print(f"   Device features: {len(device_features)}")
    print(f"   Action probs shape: {action_probs.shape}")
    print(f"   State value: {state_value.item():.3f}")
    print("   ✅ IoT Aggregation Agent working")

    # Test 4: MAPPO Agent
    print("\n4️⃣ Testing MAPPO Caching/Offloading Agent...")
    mappo_agent = MAPPOCachingOffloadingAgent(temporal_dim=64, spatial_dim=64)

    # Sample task burst
    task_burst = [
        {'required_cpu': 5, 'delay_bound': 10.0, 'size': 2.0},
        {'required_cpu': 8, 'delay_bound': 15.0, 'size': 1.5},
        {'required_cpu': 12, 'delay_bound': 20.0, 'size': 3.0}
    ]

    task_features = mappo_agent.create_task_burst_features(task_burst, 0.8, 0.3)
    print(f"   Task burst size: {len(task_burst)}")
    print(f"   Task features shape: {task_features.shape}")

    # Test offloading
    spatial_embedding = torch.randn(1, 64)
    offload_probs = mappo_agent.forward_offloading(temporal_embedding, spatial_embedding, task_features)
    print(f"   Offload probs shape: {offload_probs.shape}")

    # Test caching
    content_item = {'size': 5.0, 'ttl': 1200, 'usefulness': 0.8, 'origin': 0}
    content_features = mappo_agent.create_content_features(content_item)
    cache_prob = mappo_agent.forward_caching(temporal_embedding, spatial_embedding, content_features.unsqueeze(0))
    print(f"   Cache prob: {cache_prob.item():.3f}")

    # Test critic
    value = mappo_agent.forward_critic(temporal_embedding, spatial_embedding)
    print(f"   Value estimate: {value.item():.3f}")
    print("   ✅ MAPPO Agent working")

    # Test 5: Centralized OFDM Agent
    print("\n5️⃣ Testing Centralized OFDM Agent...")
    ofdm_agent = CentralizedOFDMAgent(max_uavs=4, max_ofdm_slots=3)

    # Sample global state
    global_state = torch.randn(1, 16)  # 4 UAVs × 4 features

    logits, state_value = ofdm_agent(global_state)
    actions = ofdm_agent.sample_actions(logits, num_slots=3)

    print(f"   Global state shape: {global_state.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Selected UAVs: {actions.sum().item()}")
    print("   ✅ OFDM Agent working")

    print("\n🎉 All individual components working correctly!")
    return True


def main():
    """Main test execution"""
    print("🚀 SAGIN Hierarchical RL Component Testing")
    print("=" * 60)

    try:
        # Test individual components first
        test_individual_components()

        print("\n" + "=" * 60)

        # Test integrated hierarchical agent
        test_hierarchical_agent()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ RL components are ready for integration with SAGIN environment")
        print("🚀 You can now run the full comparison: python rl_baseline_comparison.py")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()