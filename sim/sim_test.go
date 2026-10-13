package sim_test

import (
	"fmt"
	"math"
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

// TestAdmissionModeBreakVsSkip verifies that Break halts admission while Skip allows smaller requests to bypass.
func TestAdmissionModeBreakVsSkip(t *testing.T) {
	serverCfg := sim.ServerConfig{
		KVCapacityTokens: 500,
		MaxBatchSize:     10,
		BaseStepTimeUs:   1000,
		PerTokenCostUs:   1,
	}

	// r1 (running): 300 input tokens -> takes 300 KV.
	// rLarge (tenant-large, arrived at 10): 250 input tokens -> needs 251 KV, cannot fit in remaining 200 KV!
	// rSmall (tenant-small, arrived at 20): 50 input tokens -> needs 51 KV, would fit in remaining 200 KV!
	r1 := sim.NewRequest("r1", "t0", 300, 50, 0, 0)
	rLarge := sim.NewRequest("r-large", "t-large", 250, 20, 0, 10)
	rSmall := sim.NewRequest("r-small", "t-small", 50, 10, 0, 20)

	// Mode 1: Break (default)
	{
		simBreak := sim.NewSimulator(serverCfg, policy.NewFIFO(), 500)
		simBreak.SetAdmissionMode(sim.AdmissionModeBreak)
		trace := []*sim.Request{r1, rLarge, rSmall}
		simBreak.Run(trace)

		// Under Break, r-large must be dispatched BEFORE r-small
		if rSmall.DispatchedTime() < rLarge.DispatchedTime() {
			t.Fatalf("under AdmissionModeBreak, rSmall should NOT bypass rLarge! (small: %d, large: %d)",
				rSmall.DispatchedTime(), rLarge.DispatchedTime())
		}
	}

	// Mode 2: Skip
	{
		r1B := sim.NewRequest("r1", "t0", 300, 50, 0, 0)
		rLargeB := sim.NewRequest("r-large", "t-large", 250, 20, 0, 10)
		rSmallB := sim.NewRequest("r-small", "t-small", 50, 10, 0, 20)

		simSkip := sim.NewSimulator(serverCfg, policy.NewFIFO(), 500)
		simSkip.SetAdmissionMode(sim.AdmissionModeSkip)
		trace := []*sim.Request{r1B, rLargeB, rSmallB}
		simSkip.Run(trace)

		// Under Skip, rSmall CAN bypass rLarge while rLarge cannot fit!
		if rSmallB.DispatchedTime() >= rLargeB.DispatchedTime() {
			t.Fatalf("under AdmissionModeSkip, rSmall should bypass non-fitting rLarge! (small: %d, large: %d)",
				rSmallB.DispatchedTime(), rLargeB.DispatchedTime())
		}
	}
}

// TestLargeRequestStarvationUnderSkip proves that Skip mode can starve a large request indefinitely
// when small requests continually occupy the KV cache.
func TestLargeRequestStarvationUnderSkip(t *testing.T) {
	serverCfg := sim.ServerConfig{
		KVCapacityTokens: 500,
		MaxBatchSize:     10,
		BaseStepTimeUs:   1000,
		PerTokenCostUs:   1,
	}

	simSkip := sim.NewSimulator(serverCfg, policy.NewFIFO(), 500)
	simSkip.SetAdmissionMode(sim.AdmissionModeSkip)

	// rBase occupies 200 tokens initially (from t=0 to t=100000).
	rBase := sim.NewRequest("r-base", "tenant-base", 200, 100, 0, 0)

	// Large request arrives at t=10, requires 400 tokens out of 500.
	// Cannot fit because 200 + 401 = 601 > 500!
	rLarge := sim.NewRequest("r-large", "tenant-large", 400, 10, 0, 10)

	// Stream of small requests (each needing 100 tokens).
	// Arrive at t=20, 40, 60...
	// While rBase runs, free KV is 300.
	// rLarge (needs 400) CANNOT fit, but small requests (need 100) fit and admit immediately!
	var trace []*sim.Request
	trace = append(trace, rBase, rLarge)
	for i := 0; i < 5; i++ {
		rSmall := sim.NewRequest(fmt.Sprintf("r-small-%d", i), fmt.Sprintf("tenant-small-%d", i), 100, 10, 0, int64(20+i*20))
		trace = append(trace, rSmall)
	}

	simSkip.Run(trace)

	// Under Skip mode, all small requests bypass the earlier-arrived large request!
	for i := 0; i < 5; i++ {
		smallReq := trace[2+i]
		if smallReq.DispatchedTime() >= rLarge.DispatchedTime() {
			t.Fatalf("expected small request %s (dispatched at %d) to bypass large request (dispatched at %d)",
				smallReq.ID(), smallReq.DispatchedTime(), rLarge.DispatchedTime())
		}
	}
}

// TestJainsDemandCappedHandComputed verifies the exact mathematical formula for time-windowed demand-capped Jain index.
func TestJainsDemandCappedHandComputed(t *testing.T) {
	// Case 1: Perfectly equal service and demand
	// Both tenants demand 500, served 500. Fair share = 500. Targets = 500. x = [1.0, 1.0] -> J = 1.0
	j1 := sim.JainsDemandCapped([]float64{500, 500}, []float64{500, 500})
	if j1 != 1.0 {
		t.Fatalf("expected J=1.0 for equal allocations, got %f", j1)
	}

	// Case 2: Unequal demand, demand-capped satisfied
	// Total served = 1000, N=2 -> Fair share = 500.
	// Tenant A demand = 200 < 500 -> Target A = 200. Served A = 200 -> x_A = 1.0.
	// Tenant B demand = 800 > 500 -> Target B = 500. Served B = 800 -> x_B = 800/500 = 1.6.
	// x = [1.0, 1.6]. Sum = 2.6, SumSq = 1 + 2.56 = 3.56.
	// J = (2.6^2) / (2 * 3.56) = 6.76 / 7.12 = 0.9494382...
	j2 := sim.JainsDemandCapped([]float64{200, 800}, []float64{200, 800})
	expectedJ2 := 6.76 / 7.12
	if math.Abs(j2-expectedJ2) > 1e-6 {
		t.Fatalf("expected J=%f, got %f", expectedJ2, j2)
	}

	// Case 3: Starvation (light tenant starved below its demand)
	// Total served = 1000, N=2 -> Fair share = 500.
	// Tenant A demand = 200 -> Target A = 200. Served A = 50 -> x_A = 50/200 = 0.25.
	// Tenant B demand = 800 -> Target B = 500. Served B = 950 -> x_B = 950/500 = 1.9.
	// x = [0.25, 1.9]. Sum = 2.15, SumSq = 0.0625 + 3.61 = 3.6725.
	// J = (2.15^2) / (2 * 3.6725) = 4.6225 / 7.345 = 0.629339...
	j3 := sim.JainsDemandCapped([]float64{50, 950}, []float64{200, 800})
	expectedJ3 := 4.6225 / 7.345
	if math.Abs(j3-expectedJ3) > 1e-6 {
		t.Fatalf("expected J=%f, got %f", expectedJ3, j3)
	}
}

