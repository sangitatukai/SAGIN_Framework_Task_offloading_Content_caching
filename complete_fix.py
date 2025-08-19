# complete_fix.py - Force reload and test the fixed RL formulation
import sys
import importlib
import torch
import numpy as np

# Force remove cached modules
modules_to_remove = []
for module_name in list(sys.modules.keys()):
    if 'rl_formulation_sagin' in module_name or 'simple_rl_test' in module_name:
        modules_to_remove.append(module_name)

for module_name in modules_to_remove:
    del sys.modules[module_name]
    print(f"Removed cached module: {module_name}")

# Now import the fixed version
try:
    from rl_formulation_sagin import (
        GRUTemporalEncoder,
        GraphNeuralNetworkEncoder,
        IoTAggregationAgent,
        MAPPOCachingOffloadingAgent,
        CentralizedOFDMAgent,
        HierarchicalSAGINAgent,
        test_hierarchical_agent
    )

    print("✅ Successfully imported fixed RL modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you've replaced rl_formulation_sagin.py with the fixed version")
    sys.exit(1)


def test_individual_components():
    """Test each component to verify they work correctly"""
    print("🔬 Testing Individual Components")
    print("=" * 50)

    # Test 1: GRU Temporal Encoder
    print("1️⃣ Testing GRU Temporal Encoder...")
    try:
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
        assert obs.shape[0] == 16, f"Expected 16 elements, got {obs.shape[0]}"

        # Test forward pass
        temporal_embedding = temporal_encoder(obs)
        print(f"   Temporal embedding shape: {temporal_embedding.shape}")
        assert temporal_embedding.shape == (1, 64), f"Expected (1, 64), got {temporal_embedding.shape}"

        print("   ✅ GRU Temporal Encoder working correctly!")

    except Exception as e:
        print(f"   ❌ GRU Temporal Encoder failed: {e}")
        return False

    # Test 2: Hierarchical Agent
    print("\n2️⃣ Testing Hierarchical Agent Creation...")
    try:
        agent = HierarchicalSAGINAgent(grid_size=(2, 2))
        print(f"   Agent created with {sum(p.numel() for p in agent.parameters()):,} parameters")
        print("   ✅ Hierarchical Agent creation successful!")
    except Exception as e:
        print(f"   ❌ Hierarchical Agent creation failed: {e}")
        return False

    # Test 3: IoT Aggregation Step
    print("\n3️⃣ Testing IoT Aggregation Step...")
    try:
        # Test data
        uav_states = {
            (0, 0): {'zipf_param': 1.8, 'energy_ratio': 0.9, 'queue_ratio': 0.1, 'cache_hit_rate': 0.6},
            (0, 1): {'zipf_param': 2.1, 'energy_ratio': 0.8, 'queue_ratio': 0.3, 'cache_hit_rate': 0.4},
            (1, 0): {'zipf_param': 1.5, 'energy_ratio': 0.7, 'queue_ratio': 0.2, 'cache_hit_rate': 0.5},
            (1, 1): {'zipf_param': 1.9, 'energy_ratio': 0.6, 'queue_ratio': 0.4, 'cache_hit_rate': 0.3}
        }

        active_devices = {
            (0, 0): [1, 3, 5],
            (0, 1): [2, 4],
            (1, 0): [1, 2, 6],
            (1, 1): [3, 7, 8]
        }

        device_contents = {
            (0, 0): {1: {'size': 5.0, 'ttl': 1200}, 3: {'size': 8.0, 'ttl': 900}, 5: {'size': 3.0, 'ttl': 1500}},
            (0, 1): {2: {'size': 4.0, 'ttl': 1000}, 4: {'size': 6.0, 'ttl': 800}},
            (1, 0): {1: {'size': 7.0, 'ttl': 1100}, 2: {'size': 2.0, 'ttl': 1300}, 6: {'size': 9.0, 'ttl': 700}},
            (1, 1): {3: {'size': 5.5, 'ttl': 950}, 7: {'size': 3.5, 'ttl': 1400}, 8: {'size': 4.5, 'ttl': 1050}}
        }

        selected_devices = agent.step_iot_aggregation(uav_states, active_devices, device_contents)
        print(f"   Selected devices: {selected_devices}")
        print("   ✅ IoT Aggregation step successful!")

    except Exception as e:
        print(f"   ❌ IoT Aggregation step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n🎉 All individual components working correctly!")
    return True


def run_full_test():
    """Run the complete test"""
    print("🧪 Running Complete Test with Fixed RL Formulation")
    print("=" * 60)

    try:
        # Test individual components first
        if not test_individual_components():
            print("❌ Individual component test failed")
            return False

        print("\n" + "=" * 60)
        print("4️⃣ Running Full Hierarchical Agent Test...")

        # Run the full test
        test_agent = test_hierarchical_agent()

        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The RL formulation is now working correctly!")
        print("🚀 You can proceed with running your SAGIN simulations!")

        return True

    except Exception as e:
        print(f"❌ Full test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 SAGIN RL Formulation Complete Fix and Test")
    print("=" * 70)

    success = run_full_test()

    if success:
        print("\n✅ Fix successful! Your RL formulation is ready to use.")
        print("📝 Next steps:")
        print("   1. You can now run: python rl_sagin_integration.py")
        print("   2. Or run: python main_baseline_comparison.py")
        print("   3. The hierarchical RL agent should work without errors")
    else:
        print("\n❌ Fix failed. Please check the error messages above.")
        print("📝 Troubleshooting:")
        print("   1. Make sure you replaced rl_formulation_sagin.py with the fixed version")
        print("   2. Check that all required dependencies are installed")
        print("   3. Restart your Python environment if needed")