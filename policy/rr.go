package policy

import "sync"

// RRAdvanceMode specifies when the Round-Robin flow selection cursor advances.
type RRAdvanceMode int

const (
	// RRAdvanceUpstream models upstream llm-d-router: the cursor advances when Pick() is called.
	// When a candidate request fails engine admission (e.g. KV cache headroom), the cursor
	// has already advanced, allowing subsequent cycles to evaluate other backlogged flows.
	RRAdvanceUpstream RRAdvanceMode = iota

	// RRAdvanceIdealized models idealized 1:1 request round-robin: the cursor advances strictly
	// when OnDispatch() is confirmed by the engine. If a candidate fails admission, the cursor
	// remains on that flow, enforcing strict cyclic turns regardless of engine KV headroom.
	RRAdvanceIdealized

	// Backward-compatible aliases:
	RRAdvanceOnPick     = RRAdvanceUpstream
	RRAdvanceOnDispatch = RRAdvanceIdealized
)

func (m RRAdvanceMode) String() string {
	switch m {
	case RRAdvanceUpstream:
		return "upstream"
	case RRAdvanceIdealized:
		return "idealized"
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
	return &RoundRobin{advanceMode: RRAdvanceUpstream}
}

func NewRoundRobinUpstream() *RoundRobin {
	return &RoundRobin{advanceMode: RRAdvanceUpstream}
}

func NewRoundRobinIdealized() *RoundRobin {
	return &RoundRobin{advanceMode: RRAdvanceIdealized}
}

func NewRoundRobinWithAdvance(mode RRAdvanceMode) *RoundRobin {
	return &RoundRobin{advanceMode: mode}
}

func (p *RoundRobin) Name() string {
	if p.advanceMode == RRAdvanceIdealized {
		return "round-robin-idealized"
	}
	return "round-robin-upstream"
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
	if p.advanceMode == RRAdvanceUpstream {
		p.lastSelected = chosen
	}
	return chosen
}

func (p *RoundRobin) OnDispatch(req RequestView) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.advanceMode == RRAdvanceIdealized {
		p.lastSelected = req.TenantID()
	}
}

func (p *RoundRobin) OnArrival(req RequestView, queueWasEmpty bool, lastDepartedTenant string, otherBackloggedTenants []string) {
}

func (p *RoundRobin) OnTokens(tenantID string, inputTokens, outputTokens int)                  {}
func (p *RoundRobin) OnComplete(tenantID string, req RequestView, totalInput, totalOutput int) {}
