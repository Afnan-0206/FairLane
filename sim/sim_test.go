package sim_test

import (
	"fmt"
	"testing"

	"fairlane/policy"
	"fairlane/sim"
)

// TestDeterminism verifies that two runs with the exact same seed produce identical results.
func TestDeterminism(t *testing.T) {
	const seed int64 = 42

	configs := []sim.TenantWorkloadConfig{
		{
			TenantID:        "tenant-victim",
			RequestCount:    50,
			ArrivalProcess:  "poisson",
			MeanIntervalUs:  20000,
			PromptLengthMin: 128,
			PromptLengthMax: 256,
			OutputLengthMin: 32,
			OutputLengthMax: 64,
		},
		{
			TenantID:        "tenant-flooder",
			RequestCount:    100,
			ArrivalProcess:  "poisson",
			MeanIntervalUs:  10000,
			PromptLengthMin: 512,
			PromptLengthMax: 1024,
			OutputLengthMin: 128,
			OutputLengthMax: 256,
		},
	}

	gen1 := sim.NewWorkloadGenerator(seed)
	trace1 := gen1.GenerateTrace(configs)

	gen2 := sim.NewWorkloadGenerator(seed)
	trace2 := gen2.GenerateTrace(configs)

	if len(trace1) != len(trace2) {
		t.Fatalf("traces differ in length: %d vs %d", len(trace1), len(trace2))
	}

	for i := range trace1 {
		if trace1[i].ID() != trace2[i].ID() ||
			trace1[i].InputTokens() != trace2[i].InputTokens() ||
			trace1[i].ActualOutputTokens() != trace2[i].ActualOutputTokens() ||
			trace1[i].ArrivalTime() != trace2[i].ArrivalTime() {
			t.Fatalf("traces differ at index %d", i)
		}
	}

	// Run simulator twice with VTC
	serverCfg := sim.DefaultServerConfig()

	sim1 := sim.NewSimulator(serverCfg, policy.NewVTC(policy.DefaultVTCConfig()), 1000)
	metrics1 := sim1.Run(trace1)

	sim2 := sim.NewSimulator(serverCfg, policy.NewVTC(policy.DefaultVTCConfig()), 1000)
	metrics2 := sim2.Run(trace2)

	if metrics1.TotalVirtualTimeUs != metrics2.TotalVirtualTimeUs {
		t.Fatalf("virtual time mismatch: %d vs %d", metrics1.TotalVirtualTimeUs, metrics2.TotalVirtualTimeUs)
	}

	if metrics1.TotalTokensServed != metrics2.TotalTokensServed {
		t.Fatalf("tokens served mismatch: %d vs %d", metrics1.TotalTokensServed, metrics2.TotalTokensServed)
	}

	for tid, tm1 := range metrics1.PerTenant {
		tm2 := metrics2.PerTenant[tid]
		if tm1.TotalTokensServed != tm2.TotalTokensServed ||
			tm1.RequestsCompleted != tm2.RequestsCompleted ||
			tm1.TTFTUs.Mean != tm2.TTFTUs.Mean {
			t.Fatalf("tenant %s metrics differ across runs!", tid)
		}
	}
}

// TestTokensConservation verifies that tokens in = tokens served and no request is duplicated or lost.
func TestTokensConservation(t *testing.T) {
	const seed int64 = 12345
	configs := []sim.TenantWorkloadConfig{
		{
			TenantID:        "t1",
			RequestCount:    20,
			ArrivalProcess:  "periodic",
			MeanIntervalUs:  5000,
			PromptLengthMin: 100,
			PromptLengthMax: 100,
			OutputLengthMin: 50,
			OutputLengthMax: 50,
		},
		{
			TenantID:        "t2",
			RequestCount:    30,
			ArrivalProcess:  "periodic",
			MeanIntervalUs:  4000,
			PromptLengthMin: 200,
			PromptLengthMax: 200,
			OutputLengthMin: 80,
			OutputLengthMax: 80,
		},
	}

	trace := sim.NewWorkloadGenerator(seed).GenerateTrace(configs)

	expectedTotalTokens := 0
	expectedReqCount := len(trace)
	for _, r := range trace {
		expectedTotalTokens += r.InputTokens() + r.ActualOutputTokens()
	}

	serverCfg := sim.DefaultServerConfig()
	simulator := sim.NewSimulator(serverCfg, policy.NewFIFO(), 500)
	metrics := simulator.Run(trace)

	if metrics.TotalCompletedReqs != expectedReqCount {
		t.Fatalf("expected %d completed requests, got %d", expectedReqCount, metrics.TotalCompletedReqs)
	}

	if metrics.TotalTokensServed != expectedTotalTokens {
		t.Fatalf("expected %d total tokens served, got %d", expectedTotalTokens, metrics.TotalTokensServed)
	}
}

// TestWorkConservationInvariant ensures that idle-while-queued time is strictly 0.
func TestWorkConservationInvariant(t *testing.T) {
	configs := []sim.TenantWorkloadConfig{
		{
			TenantID:        "t1",
			RequestCount:    30,
			ArrivalProcess:  "poisson",
			MeanIntervalUs:  10000,
			PromptLengthMin: 64,
			PromptLengthMax: 128,
			OutputLengthMin: 32,
			OutputLengthMax: 64,
		},
	}

	policies := []policy.Policy{
		policy.NewFIFO(),
		policy.NewRoundRobin(),
		policy.NewVTC(policy.DefaultVTCConfig()),
	}

	serverCfg := sim.DefaultServerConfig()

	for _, p := range policies {
		trace := sim.NewWorkloadGenerator(999).GenerateTrace(configs)
		simulator := sim.NewSimulator(serverCfg, p, 1000)
		metrics := simulator.Run(trace)

		if metrics.IdleWhileQueuedUs != 0 {
			t.Fatalf("WORK-CONSERVATION VIOLATION: policy %s had idle-while-queued time = %d µs (expected 0)",
				p.Name(), metrics.IdleWhileQueuedUs)
		}
	}
}

// TestBreakOnNoFit verifies that when winning tenant cannot fit, the server breaks and does not bypass.
func TestBreakOnNoFit(t *testing.T) {
	// Server with small KV capacity: 500 tokens
	serverCfg := sim.ServerConfig{
		KVCapacityTokens: 500,
		MaxBatchSize:     10,
		BaseStepTimeUs:   1000,
		PerTokenCostUs:   1,
	}

	pol := policy.NewFIFO()
	simulator := sim.NewSimulator(serverCfg, pol, 500)

	// r1 arrives at 0: 300 input, 100 output (takes 300 KV).
	// r2 (winning tenant) arrives at 10: 250 input (cannot fit in remaining 200 KV).
	// r3 (other tenant) arrives at 20: 50 input (would fit in 200 KV, but must NOT bypass r2!).
	r1 := sim.NewRequest("r1", "t1", 300, 50, 0, 0)
	r2 := sim.NewRequest("r2", "t2", 250, 20, 0, 10)
	r3 := sim.NewRequest("r3", "t3", 50, 10, 0, 20)

	trace := []*sim.Request{r1, r2, r3}
	metrics := simulator.Run(trace)

	if metrics.TotalCompletedReqs != 3 {
		t.Fatalf("expected all 3 to complete eventually, got %d", metrics.TotalCompletedReqs)
	}

	// r2 must be dispatched BEFORE r3 under FIFO, even though r3 could have fit earlier
	if r2.DispatchedTime() > r3.DispatchedTime() {
		t.Fatalf("BREAK-ON-NO-FIT VIOLATION: r3 was dispatched at %d before r2 at %d (bypass occurred)",
			r3.DispatchedTime(), r2.DispatchedTime())
	}
}

// TestNeverFitsRejection verifies that requests larger than total capacity are rejected immediately.
func TestNeverFitsRejection(t *testing.T) {
	serverCfg := sim.ServerConfig{
		KVCapacityTokens: 500,
		MaxBatchSize:     10,
		BaseStepTimeUs:   1000,
		PerTokenCostUs:   1,
	}

	simulator := sim.NewSimulator(serverCfg, policy.NewFIFO(), 500)

	rValid := sim.NewRequest("r-valid", "t1", 200, 20, 0, 0)
	rOversize := sim.NewRequest("r-oversize", "t1", 600, 20, 0, 10) // 600 > 500: NEVER FITS!

	trace := []*sim.Request{rValid, rOversize}
	metrics := simulator.Run(trace)

	if metrics.NeverFitsRejected != 1 {
		t.Fatalf("expected 1 never-fits rejected, got %d", metrics.NeverFitsRejected)
	}

	if metrics.TotalCompletedReqs != 1 {
		t.Fatalf("expected 1 completed request, got %d", metrics.TotalCompletedReqs)
	}
}

// TestSameTenantRejoin verifies counter behavior when a tenant's queue becomes empty and rejoins.
func TestSameTenantRejoin(t *testing.T) {
	serverCfg := sim.DefaultServerConfig()

	for _, mode := range []policy.LiftMode{policy.LiftModePaper, policy.LiftModeArtifact} {
		t.Run(fmt.Sprintf("mode-%s", mode), func(t *testing.T) {
			cfg := policy.DefaultVTCConfig()
			cfg.LiftMode = mode
			pol := policy.NewVTC(cfg)

			simulator := sim.NewSimulator(serverCfg, pol, 500)

			// Tenant A submits r1, finishes, queue becomes empty.
			// System goes idle. Then Tenant A submits r2 (same tenant rejoin).
			r1 := sim.NewRequest("r1", "tenant-A", 100, 50, 0, 0)
			r2 := sim.NewRequest("r2", "tenant-A", 100, 50, 0, 500000)

			trace := []*sim.Request{r1, r2}
			metrics := simulator.Run(trace)

			if metrics.TotalCompletedReqs != 2 {
				t.Fatalf("expected 2 completed requests, got %d", metrics.TotalCompletedReqs)
			}
		})
	}
}
