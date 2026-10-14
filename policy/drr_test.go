package policy_test

import (
	"testing"

	"fairlane/policy"
)

// TestDRRHandComputedOracle tests exact deficit round robin accumulation and token parity.
func TestDRRHandComputedOracle(t *testing.T) {
	cfg := policy.DRRConfig{
		BaseQuantumTokens: 500,
		TenantWeights:     map[string]float64{"tenant-a": 1.0, "tenant-b": 1.0},
	}
	drr := policy.NewDRR(cfg)

	// Tenant A has requests of cost 100; Tenant B has requests of cost 400.
	reqsA := []*mockRequest{
		{id: "a-1", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		{id: "a-2", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		{id: "a-3", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		{id: "a-4", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		{id: "a-5", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		{id: "a-6", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		{id: "a-7", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		{id: "a-8", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		{id: "a-9", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		{id: "a-10", tenantID: "tenant-a", input: 100, arrivalUs: 0},
	}
	reqsB := []*mockRequest{
		{id: "b-1", tenantID: "tenant-b", input: 400, arrivalUs: 0},
		{id: "b-2", tenantID: "tenant-b", input: 400, arrivalUs: 0},
		{id: "b-3", tenantID: "tenant-b", input: 400, arrivalUs: 0},
	}

	idxA, idxB := 0, 0
	tokensServedA, tokensServedB := 0, 0

	// Run scheduling turns until B completes 2 requests
	for idxB < 2 {
		heads := make(map[string]policy.RequestView)
		if idxA < len(reqsA) {
			heads["tenant-a"] = reqsA[idxA]
		}
		if idxB < len(reqsB) {
			heads["tenant-b"] = reqsB[idxB]
		}
		if len(heads) == 0 {
			break
		}

		view := newMockView(heads, 100)
		winner := drr.Pick(view)
		if winner == "" {
			break
		}

		if winner == "tenant-a" {
			req := reqsA[idxA]
			drr.OnDispatch(req)
			tokensServedA += req.InputTokens()
			idxA++
		} else {
			req := reqsB[idxB]
			drr.OnDispatch(req)
			tokensServedB += req.InputTokens()
			idxB++
		}
	}

	// In DRR:
	// Round 1: A dispatches 5 reqs (500 tokens), deficit A = 0.
	//          B dispatches 1 req (400 tokens), deficit B = 100.
	// Round 2: A gets 500 -> dispatches 5 reqs (500 tokens).
	//          B gets 500 -> deficit B = 600 -> dispatches 1 req (400 tokens), deficit B = 200.
	// Total: A dispatches 10 reqs = 1000 tokens, B dispatches 2 reqs = 800 tokens.
	if tokensServedA != 1000 {
		t.Fatalf("Expected 1000 tokens for tenant-a, got %d", tokensServedA)
	}
	if tokensServedB != 800 {
		t.Fatalf("Expected 800 tokens for tenant-b, got %d", tokensServedB)
	}
}

// TestDRRWeightedOracle tests unequal weights (w_A = 2.0, w_B = 1.0)
func TestDRRWeightedOracle(t *testing.T) {
	cfg := policy.DRRConfig{
		BaseQuantumTokens: 200,
		TenantWeights:     map[string]float64{"tenant-a": 2.0, "tenant-b": 1.0}, // Q_A = 400, Q_B = 200
	}
	drr := policy.NewDRR(cfg)

	heads := map[string]policy.RequestView{
		"tenant-a": &mockRequest{id: "a-1", tenantID: "tenant-a", input: 100, arrivalUs: 0},
		"tenant-b": &mockRequest{id: "b-1", tenantID: "tenant-b", input: 100, arrivalUs: 0},
	}
	view := newMockView(heads, 100)

	// A has quantum 400 -> can serve 4 requests of 100
	// B has quantum 200 -> can serve 2 requests of 100
	servesA, servesB := 0, 0
	for i := 0; i < 6; i++ {
		winner := drr.Pick(view)
		drr.OnDispatch(heads[winner])
		if winner == "tenant-a" {
			servesA++
		} else {
			servesB++
		}
	}

	if servesA != 4 || servesB != 2 {
		t.Fatalf("Expected 4 serves for A and 2 serves for B (2:1 ratio), got A=%d, B=%d", servesA, servesB)
	}
}

// TestDRRCompletionCorrection tests that underpredicted output tokens are charged upon completion.
func TestDRRCompletionCorrection(t *testing.T) {
	cfg := policy.DRRConfig{
		BaseQuantumTokens: 500,
		InputWeight:       1.0,
		OutputWeight:      2.0,
		UsePredictedCost:  true,
	}
	drr := policy.NewDRR(cfg)

	req := &mockRequest{
		id:        "req-1",
		tenantID:  "tenant-a",
		input:     100,
		predOut:   50, // predicted cost = 100 + 2*50 = 200
		arrivalUs: 0,
	}

	heads := map[string]policy.RequestView{"tenant-a": req}
	view := newMockView(heads, 100)

	winner := drr.Pick(view)
	if winner != "tenant-a" {
		t.Fatalf("Expected tenant-a to be picked, got %s", winner)
	}

	drr.OnDispatch(req)
	// Deficit should be 500 - 200 = 300
	deficits := drr.Deficits()
	if deficits["tenant-a"] != 300.0 {
		t.Fatalf("Expected deficit 300 after dispatch, got %f", deficits["tenant-a"])
	}

	// Request completes with actual output 150 (diff = +100 tokens, weighted delta = +200)
	drr.OnComplete("tenant-a", req, 100, 150)
	deficits = drr.Deficits()
	if deficits["tenant-a"] != 100.0 {
		t.Fatalf("Expected deficit 100 after completion correction, got %f", deficits["tenant-a"])
	}
}

