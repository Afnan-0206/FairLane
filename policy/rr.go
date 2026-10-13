package policy

import "sync"

// RRAdvanceMode specifies when the Round-Robin flow selection cursor advances.
type RRAdvanceMode int

const (
	// RRAdvanceOnPick advances the cursor immediately when Pick() is called (Default).
	// Bypasses tenants whose head requests cannot currently fit into the batch/KV cache.
	RRAdvanceOnPick RRAdvanceMode = iota

	// RRAdvanceOnDispatch advances the cursor only when OnDispatch() is confirmed by the server.
	// Preserves strict cyclic turns even when a head request fails admission under break-on-no-fit.
	RRAdvanceOnDispatch
)

func (m RRAdvanceMode) String() string {
	switch m {
	case RRAdvanceOnPick:
		return "onPick"
	case RRAdvanceOnDispatch:
		return "onDispatch"
	default:
		return "unknown"
	}
}

// RoundRobin implements per-tenant request-count round-robin flow selection.
// It maps directly to llm-d-router's round-robin-fairness-policy.
type RoundRobin struct {
	mu           sync.Mutex
	advanceMode  RRAdvanceMode
	lastSelected string
	candidate    string
}

func NewRoundRobin() *RoundRobin {
	return &RoundRobin{advanceMode: RRAdvanceOnPick}
}

func NewRoundRobinWithAdvance(mode RRAdvanceMode) *RoundRobin {
	return &RoundRobin{advanceMode: mode}
}

func (p *RoundRobin) Name() string {
	if p.advanceMode == RRAdvanceOnDispatch {
		return "round-robin (onDispatch)"
	}
	return "round-robin"
}

func (p *RoundRobin) AdvanceMode() RRAdvanceMode {
	return p.advanceMode
}

func (p *RoundRobin) Pick(view SchedulerView) string {
	p.mu.Lock()
	defer p.mu.Unlock()

	tenants := view.BackloggedTenants()
	if len(tenants) == 0 {
		return ""
	}

	var chosen string
	// If no previous selection, pick the first tenant
	if p.lastSelected == "" {
		chosen = tenants[0]
	} else {
		// Search for the position of lastSelected in the current sorted backlogged tenants
		idx := -1
		for i, t := range tenants {
			if t == p.lastSelected {
				idx = i
				break
			}
		}

		if idx != -1 {
			// Pick next in cyclic order
			nextIdx := (idx + 1) % len(tenants)
			chosen = tenants[nextIdx]
		} else {
			// If lastSelected is no longer backlogged, pick the first tenant lexicographically
			// greater than lastSelected, otherwise wrap around to index 0.
			chosen = tenants[0]
			for _, t := range tenants {
				if t > p.lastSelected {
					chosen = t
					break
				}
			}
		}
	}

	p.candidate = chosen
	if p.advanceMode == RRAdvanceOnPick {
		p.lastSelected = chosen
	}
	return chosen
}

func (p *RoundRobin) OnDispatch(req RequestView) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.advanceMode == RRAdvanceOnDispatch {
		p.lastSelected = req.TenantID()
	}
}

func (p *RoundRobin) OnArrival(req RequestView, queueWasEmpty bool, lastDepartedTenant string, otherBackloggedTenants []string) {
}

func (p *RoundRobin) OnTokens(tenantID string, inputTokens, outputTokens int)                  {}
func (p *RoundRobin) OnComplete(tenantID string, req RequestView, totalInput, totalOutput int) {}
