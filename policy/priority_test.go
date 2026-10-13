package policy_test

import (
	"testing"

	"fairlane/policy"
)

// TestPriorityBandsStrictInterTier tests that Tier 0 always preempts Tier 1.
func TestPriorityBandsStrictInterTier(t *testing.T) {
	subPolicies := map[int]policy.Policy{
		0: policy.NewFIFO(),
		1: policy.NewFIFO(),
	}
	p := policy.NewPriorityBandPolicy(subPolicies)

	// High priority tenant and Low priority tenant
	reqHigh := &mockRequest{id: "high-1", tenantID: "tenant-high", input: 100, priority: 0}
	reqLow := &mockRequest{id: "low-1", tenantID: "tenant-low", input: 100, priority: 1}

	heads := map[string]policy.RequestView{
		"tenant-high": reqHigh,
		"tenant-low":  reqLow,
	}
	view := newMockView(heads, 100)

	winner := p.Pick(view)
	if winner != "tenant-high" {
		t.Fatalf("Expected tenant-high to win under strict priority, got %s", winner)
	}

	// Remove high priority tenant; low priority tenant should now win
	delete(heads, "tenant-high")
	view2 := newMockView(heads, 200)

	winner2 := p.Pick(view2)
	if winner2 != "tenant-low" {
		t.Fatalf("Expected tenant-low to win when high band is empty, got %s", winner2)
	}
}

// TestPriorityBandsIntraTierFairness tests fair delegation within the same priority band.
func TestPriorityBandsIntraTierFairness(t *testing.T) {
	subPolicies := map[int]policy.Policy{
		0: policy.NewRoundRobinIdealized(),
	}
	p := policy.NewPriorityBandPolicy(subPolicies)

	reqA := &mockRequest{id: "a-1", tenantID: "tenant-a", input: 100, priority: 0}
	reqB := &mockRequest{id: "b-1", tenantID: "tenant-b", input: 100, priority: 0}

	heads := map[string]policy.RequestView{
		"tenant-a": reqA,
		"tenant-b": reqB,
	}
	view := newMockView(heads, 100)

	w1 := p.Pick(view)
	p.OnDispatch(heads[w1])
	w2 := p.Pick(view)
	p.OnDispatch(heads[w2])

	if w1 == w2 {
		t.Fatalf("Expected alternating round robin picks within priority band 0, got %s then %s", w1, w2)
	}
}
