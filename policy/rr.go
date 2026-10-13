package policy

import "sync"

// RoundRobin implements per-tenant request-count round-robin flow selection.
// It maps directly to llm-d-router's round-robin-fairness-policy:
// It advances a cursor over active flows, granting one dispatch turn per tenant in cyclic order.
type RoundRobin struct {
	mu           sync.Mutex
	lastSelected string
}

func NewRoundRobin() *RoundRobin {
	return &RoundRobin{}
}

func (p *RoundRobin) Name() string {
	return "round-robin"
}

func (p *RoundRobin) Pick(view SchedulerView) string {
	p.mu.Lock()
	defer p.mu.Unlock()

	tenants := view.BackloggedTenants()
	if len(tenants) == 0 {
		p.lastSelected = ""
		return ""
	}

	// If no previous selection or previous is empty, pick the first tenant
	if p.lastSelected == "" {
		p.lastSelected = tenants[0]
		return p.lastSelected
	}

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
		p.lastSelected = tenants[nextIdx]
		return p.lastSelected
	}

	// If lastSelected is no longer backlogged, pick the first tenant lexicographically
	// greater than lastSelected, otherwise wrap around to index 0.
	chosen := tenants[0]
	for _, t := range tenants {
		if t > p.lastSelected {
			chosen = t
			break
		}
	}
	p.lastSelected = chosen
	return p.lastSelected
}

func (p *RoundRobin) OnArrival(req RequestView, queueWasEmpty bool, lastDepartedTenant string, otherBackloggedTenants []string) {
}
func (p *RoundRobin) OnDispatch(req RequestView)                                               {}
func (p *RoundRobin) OnTokens(tenantID string, inputTokens, outputTokens int)                  {}
func (p *RoundRobin) OnComplete(tenantID string, req RequestView, totalInput, totalOutput int) {}
