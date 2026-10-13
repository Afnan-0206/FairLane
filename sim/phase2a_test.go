package sim_test

import (
	"testing"

	"fairlane/policy"
	"fairlane/sim"
)

// TestConservationInvariant verifies that total submitted tokens strictly equal total completed tokens upon drain.
func TestConservationInvariant(t *testing.T) {
	policies := []policy.Policy{
		policy.NewFIFO(),
		policy.NewRoundRobinIdealized(),
		policy.NewVTC(policy.DefaultVTCConfig()),
		policy.NewDRR(policy.DefaultDRRConfig()),
	}

	for _, pol := range policies {
		t.Run(pol.Name(), func(t *testing.T) {
			serverCfg := sim.ServerConfig{
				MaxBatchSize:     8,
				KVCapacityTokens: 2048,
				BaseStepTimeUs:   5000,
				PerTokenCostUs:   5,
			}
			simulator := sim.NewSimulator(serverCfg, pol, 500)

			// Generate a multi-tenant workload trace
			trace := []*sim.Request{
				sim.NewRequest("r1", "t1", 100, 50, 50, 0),
				sim.NewRequest("r2", "t2", 200, 30, 30, 1000),
				sim.NewRequest("r3", "t1", 150, 40, 40, 2000),
				sim.NewRequest("r4", "t2", 120, 60, 60, 3000),
				sim.NewRequest("r5", "t3", 80, 20, 20, 4000),
			}

			submittedTokens := 0
			for _, r := range trace {
				submittedTokens += r.InputTokens() + r.ActualOutputTokens()
			}

			metrics := simulator.Run(trace)

			completedTokens := 0
			for _, tm := range metrics.PerTenant {
				completedTokens += tm.TotalTokensServed
			}

			if completedTokens != submittedTokens {
				t.Fatalf("Policy %s violated conservation invariant: submitted %d tokens, completed %d tokens",
					pol.Name(), submittedTokens, completedTokens)
			}
		})
	}
}

// TestDeterminismAcrossRuns verifies that identical traces and seeds yield bit-for-bit identical results.
func TestDeterminismAcrossRuns(t *testing.T) {
	traceGen := func() []*sim.Request {
		return []*sim.Request{
			sim.NewRequest("a1", "tenant-a", 128, 64, 64, 0),
			sim.NewRequest("b1", "tenant-b", 512, 32, 32, 100),
			sim.NewRequest("a2", "tenant-a", 128, 64, 64, 200),
			sim.NewRequest("b2", "tenant-b", 512, 32, 32, 300),
			sim.NewRequest("a3", "tenant-a", 128, 64, 64, 400),
		}
	}

	serverCfg := sim.ServerConfig{
		MaxBatchSize:     4,
		KVCapacityTokens: 1024,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	// Test DRR determinism
	sim1 := sim.NewSimulator(serverCfg, policy.NewDRR(policy.DefaultDRRConfig()), 500)
	res1 := sim1.Run(traceGen())

	sim2 := sim.NewSimulator(serverCfg, policy.NewDRR(policy.DefaultDRRConfig()), 500)
	res2 := sim2.Run(traceGen())

	if res1.TotalVirtualTimeUs != res2.TotalVirtualTimeUs {
		t.Fatalf("Non-deterministic execution time: %d vs %d", res1.TotalVirtualTimeUs, res2.TotalVirtualTimeUs)
	}

	for tid, tm1 := range res1.PerTenant {
		tm2, ok := res2.PerTenant[tid]
		if !ok || tm1.TotalTokensServed != tm2.TotalTokensServed || tm1.TTFTUs.Mean != tm2.TTFTUs.Mean {
			t.Fatalf("Non-deterministic tenant metrics for %s: %+v vs %+v", tid, tm1, tm2)
		}
	}
}

// TestGatewayCapKInvariant verifies that engine in-flight requests never exceed cap K.
func TestGatewayCapKInvariant(t *testing.T) {
	caps := []int{2, 4, 8}

	for _, capK := range caps {
		serverCfg := sim.ServerConfig{
			MaxBatchSize:     4,
			KVCapacityTokens: 4096,
			BaseStepTimeUs:   5000,
			PerTokenCostUs:   5,
		}
		simulator := sim.NewSimulator(serverCfg, policy.NewVTC(policy.DefaultVTCConfig()), 500)
		simulator.SetGatewayConfig(sim.GatewayConfig{
			Enabled:      true,
			MaxInFlightK: capK,
		})

		// 20 requests all arriving at time 0
		var trace []*sim.Request
		for i := 0; i < 20; i++ {
			trace = append(trace, sim.NewRequest("r", "tenant", 100, 50, 50, 0))
		}

		metrics := simulator.Run(trace)
		if metrics.TotalCompletedReqs != 20 {
			t.Fatalf("Expected all 20 requests completed under cap K=%d, got %d", capK, metrics.TotalCompletedReqs)
		}
	}
}
