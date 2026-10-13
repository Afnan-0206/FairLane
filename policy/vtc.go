package policy

import (
	"math"
	"sync"
)

// VTCConfig holds configuration for the Virtual Token Counter policy.
type VTCConfig struct {
	InputWeight   float64
	OutputWeight  float64
	LiftMode      LiftMode
	LiftTiming    LiftTiming
	ChargingModel ChargingModel
}

// DefaultVTCConfig returns the canonical configuration verified from VTC-artifact:
// w_in = 1.0, w_out = 2.0, LiftModeArtifact, LiftTimingArrival, ChargingIncremental.
func DefaultVTCConfig() VTCConfig {
	return VTCConfig{
		InputWeight:   1.0,
		OutputWeight:  2.0,
		LiftMode:      LiftModeArtifact,
		LiftTiming:    LiftTimingArrival,
		ChargingModel: ChargingIncremental,
	}
}

// VTC implements token-fair scheduling following Sheng et al. (OSDI 2024).
// It maintains a virtual token counter per tenant and schedules the tenant with the minimum counter.
type VTC struct {
	mu     sync.Mutex
	config VTCConfig

	// counters maps tenantID to accumulated virtual tokens.
	counters map[string]float64

	// pendingLift tracks tenants that need a lift evaluation on Pick (when LiftTimingPick is active).
	pendingLift map[string]string // tenantID -> lastDepartedTenant at arrival time
}

func NewVTC(cfg VTCConfig) *VTC {
	if cfg.InputWeight <= 0 {
		cfg.InputWeight = 1.0
	}
	if cfg.OutputWeight <= 0 {
		cfg.OutputWeight = 2.0
	}
	return &VTC{
		config:      cfg,
		counters:    make(map[string]float64),
		pendingLift: make(map[string]string),
	}
}

func (p *VTC) Name() string {
	return "vtc"
}

func (p *VTC) Config() VTCConfig {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.config
}

// Counter returns a copy of the current counter for a tenant (for inspection & metrics).
func (p *VTC) Counter(tenantID string) float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.counters[tenantID]
}

// Counters returns a copy of all tenant counters.
func (p *VTC) Counters() map[string]float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	res := make(map[string]float64, len(p.counters))
	for k, v := range p.counters {
		res[k] = v
	}
	return res
}

// applyLift performs the starvation-prevention counter lift.
// It is the single unified lift function called either on arrival or lazily on pick.
//
// Invariants:
// - otherActiveTenants: slice of other active/backlogged tenant IDs (excluding tenantID).
// - lastDepartedTenant: the tenant whose queue most recently transitioned to empty.
func (p *VTC) applyLift(tenantID string, otherActiveTenants []string, lastDepartedTenant string) {
	currentVal := p.counters[tenantID]

	if len(otherActiveTenants) > 0 {
		// Normal lift: lift to the minimum of all other currently active tenant counters.
		// c_u = max(c_u, min_{j in other} c_j)
		minVal := math.MaxFloat64
		for _, other := range otherActiveTenants {
			if val, ok := p.counters[other]; ok && val < minVal {
				minVal = val
			}
		}
		if minVal != math.MaxFloat64 && minVal > currentVal {
			p.counters[tenantID] = minVal
		}
		return
	}

	// Empty-queue scenario: no other tenant is currently active in the queue.
	switch p.config.LiftMode {
	case LiftModePaper:
		// arXiv:2401.00588 Algorithm 2 lines 9-11:
		// "If Q = empty: let l <- last client left Q; c_u <- max(c_u, c_l)"
		if lastDepartedTenant != "" && lastDepartedTenant != tenantID {
			if depVal, ok := p.counters[lastDepartedTenant]; ok && depVal > currentVal {
				p.counters[tenantID] = depVal
			}
		}
	case LiftModeArtifact:
		// VTC-artifact (slora/server/router/vtc_req_queue.py lines 48-50):
		// if len(cnts) > 0: lift. If len(cnts) == 0: no-op, counter is unchanged.
		// (Do nothing).
	}
}

func (p *VTC) OnArrival(req RequestView, queueWasEmpty bool, lastDepartedTenant string, otherBackloggedTenants []string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	tenant := req.TenantID()
	if _, ok := p.counters[tenant]; !ok {
		p.counters[tenant] = 0.0
	}

	if queueWasEmpty {
		if p.config.LiftTiming == LiftTimingArrival {
			p.applyLift(tenant, otherBackloggedTenants, lastDepartedTenant)
		} else {
			// Save for lazy evaluation in Pick
			p.pendingLift[tenant] = lastDepartedTenant
		}
	}
}

func (p *VTC) Pick(view SchedulerView) string {
	p.mu.Lock()
	defer p.mu.Unlock()

	backlogged := view.BackloggedTenants()
	if len(backlogged) == 0 {
		return ""
	}

	// If LiftTimingPick is active, evaluate any pending lifts before selecting
	if p.config.LiftTiming == LiftTimingPick && len(p.pendingLift) > 0 {
		for _, t := range backlogged {
			if lastDep, ok := p.pendingLift[t]; ok {
				var otherTenants []string
				for _, other := range backlogged {
					if other != t {
						otherTenants = append(otherTenants, other)
					}
				}
				p.applyLift(t, otherTenants, lastDep)
				delete(p.pendingLift, t)
			}
		}
	}

	// Selection rule: argmin_{t in backlogged} c_t
	// Tie-break rule: lexicographically smallest tenant ID
	var bestTenant string
	bestCounter := math.MaxFloat64

	for _, t := range backlogged {
		c := p.counters[t]
		if bestTenant == "" || c < bestCounter {
			bestCounter = c
			bestTenant = t
		} else if c == bestCounter {
			if t < bestTenant {
				bestTenant = t
			}
		}
	}

	return bestTenant
}

func (p *VTC) OnDispatch(req RequestView) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.config.ChargingModel == ChargingIncremental {
		// Prefill cost charged upon admission into execution
		cost := p.config.InputWeight * float64(req.InputTokens())
		p.counters[req.TenantID()] += cost
	}
}

func (p *VTC) OnTokens(tenantID string, inputTokens, outputTokens int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.config.ChargingModel == ChargingIncremental {
		// Output tokens charged incrementally per generation step
		cost := p.config.OutputWeight * float64(outputTokens)
		p.counters[tenantID] += cost
	}
}

func (p *VTC) OnComplete(tenantID string, req RequestView, totalInput, totalOutput int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.config.ChargingModel == ChargingCompletion {
		// Full request cost charged upon completion
		cost := p.config.InputWeight*float64(totalInput) + p.config.OutputWeight*float64(totalOutput)
		p.counters[tenantID] += cost
	}
}
