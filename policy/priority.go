package policy

import (
	"sort"
	"sync"
)

// PriorityBandPolicy implements strict inter-tier priority scheduling across priority tiers,
// while delegating intra-tier scheduling to fair sub-policies (e.g. VTC, DRR, or RR).
// Priority 0 is highest priority (Tier 0), followed by 1, 2, etc.
type PriorityBandPolicy struct {
	mu           sync.Mutex
	policyPerBand map[int]Policy
	defaultBand  int
}

// NewPriorityBandPolicy constructs a priority policy with per-band delegates.
func NewPriorityBandPolicy(policyPerBand map[int]Policy) *PriorityBandPolicy {
	return &PriorityBandPolicy{
		policyPerBand: policyPerBand,
		defaultBand:   0,
	}
}

func (p *PriorityBandPolicy) Name() string {
	return "priority-bands"
}

func (p *PriorityBandPolicy) Pick(view SchedulerView) string {
	p.mu.Lock()
	defer p.mu.Unlock()

	backlogged := view.BackloggedTenants()
	if len(backlogged) == 0 {
		return ""
	}

	// Group backlogged tenants by their head request priority band
	tenantsByBand := make(map[int][]string)
	for _, t := range backlogged {
		head, ok := view.HeadRequest(t)
		if !ok || head == nil {
			continue
		}
		band := head.Priority()
		tenantsByBand[band] = append(tenantsByBand[band], t)
	}

	if len(tenantsByBand) == 0 {
		return ""
	}

	// Find the lowest priority band number (highest strict priority)
	var bands []int
	for b := range tenantsByBand {
		bands = append(bands, b)
	}
	sort.Ints(bands)
	highestBand := bands[0]

	activeTenantsInBand := tenantsByBand[highestBand]
	subPolicy, ok := p.policyPerBand[highestBand]
	if !ok || subPolicy == nil {
		// Fallback to lexicographically smallest tenant in this band
		return activeTenantsInBand[0]
	}

	// Create a sub-view containing only the tenants in this priority band
	subView := &bandFilteredView{
		parent:        view,
		allowedTenants: activeTenantsInBand,
	}

	return subPolicy.Pick(subView)
}

func (p *PriorityBandPolicy) OnArrival(req RequestView, queueWasEmpty bool, lastDepartedTenant string, otherBackloggedTenants []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	band := req.Priority()
	if sub, ok := p.policyPerBand[band]; ok && sub != nil {
		sub.OnArrival(req, queueWasEmpty, lastDepartedTenant, otherBackloggedTenants)
	}
}

func (p *PriorityBandPolicy) OnDispatch(req RequestView) {
	p.mu.Lock()
	defer p.mu.Unlock()
	band := req.Priority()
	if sub, ok := p.policyPerBand[band]; ok && sub != nil {
		sub.OnDispatch(req)
	}
}

func (p *PriorityBandPolicy) OnTokens(tenantID string, inputTokens, outputTokens int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, sub := range p.policyPerBand {
		if sub != nil {
			sub.OnTokens(tenantID, inputTokens, outputTokens)
		}
	}
}

func (p *PriorityBandPolicy) OnComplete(tenantID string, req RequestView, totalInput, totalOutput int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if req != nil {
		band := req.Priority()
		if sub, ok := p.policyPerBand[band]; ok && sub != nil {
			sub.OnComplete(tenantID, req, totalInput, totalOutput)
			return
		}
	}
	for _, sub := range p.policyPerBand {
		if sub != nil {
			sub.OnComplete(tenantID, req, totalInput, totalOutput)
		}
	}
}

// bandFilteredView exposes only tenants in the active priority band
type bandFilteredView struct {
	parent        SchedulerView
	allowedTenants []string
}

func (v *bandFilteredView) BackloggedTenants() []string {
	// Already sorted since original backlogged list was sorted
	return v.allowedTenants
}

func (v *bandFilteredView) HeadRequest(tenantID string) (RequestView, bool) {
	return v.parent.HeadRequest(tenantID)
}

func (v *bandFilteredView) CurrentTime() int64 {
	return v.parent.CurrentTime()
}
