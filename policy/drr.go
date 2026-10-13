package policy

import (
	"sync"
)

// DRRConfig configures Deficit Round Robin scheduling.
type DRRConfig struct {
	BaseQuantumTokens int                // Base token quantum Q_0 (default: 512)
	TenantWeights     map[string]float64 // Tenant weights w_i (default: 1.0)
	UsePredictedCost  bool               // If true, uses inputTokens + predictedOutput; if false, uses inputTokens
}

// DefaultDRRConfig provides baseline parameters for token-based DRR.
func DefaultDRRConfig() DRRConfig {
	return DRRConfig{
		BaseQuantumTokens: 512,
		TenantWeights:     make(map[string]float64),
		UsePredictedCost:  false,
	}
}

// DRR implements Deficit Round Robin flow selection over token costs.
type DRR struct {
	mu            sync.Mutex
	cfg           DRRConfig
	deficits      map[string]int
	currentTenant string
}

func NewDRR(cfg DRRConfig) *DRR {
	if cfg.BaseQuantumTokens <= 0 {
		cfg.BaseQuantumTokens = 512
	}
	if cfg.TenantWeights == nil {
		cfg.TenantWeights = make(map[string]float64)
	}
	return &DRR{
		cfg:      cfg,
		deficits: make(map[string]int),
	}
}

func (p *DRR) Name() string {
	return "drr"
}

func (p *DRR) Deficits() map[string]int {
	p.mu.Lock()
	defer p.mu.Unlock()
	res := make(map[string]int, len(p.deficits))
	for k, v := range p.deficits {
		res[k] = v
	}
	return res
}

func (p *DRR) quantum(tenantID string) int {
	w := 1.0
	if val, ok := p.cfg.TenantWeights[tenantID]; ok && val > 0 {
		w = val
	}
	q := int(float64(p.cfg.BaseQuantumTokens) * w)
	if q < 1 {
		q = 1
	}
	return q
}

func (p *DRR) cost(req RequestView) int {
	cost := req.InputTokens()
	if p.cfg.UsePredictedCost && req.PredictedOutputTokens() > 0 {
		cost += req.PredictedOutputTokens()
	}
	if cost < 1 {
		cost = 1
	}
	return cost
}

func (p *DRR) Pick(view SchedulerView) string {
	p.mu.Lock()
	defer p.mu.Unlock()

	tenants := view.BackloggedTenants()
	if len(tenants) == 0 {
		p.currentTenant = ""
		return ""
	}

	// Reset deficits for tenants that have no queued requests
	for t := range p.deficits {
		if _, ok := view.HeadRequest(t); !ok {
			delete(p.deficits, t)
		}
	}

	// Find index of currentTenant if present
	currIdx := -1
	if p.currentTenant != "" {
		for i, t := range tenants {
			if t == p.currentTenant {
				currIdx = i
				break
			}
		}
	}

	// If currentTenant is not set or not in current backlogged tenants,
	// pick first tenant in cyclic order after currentTenant (or tenants[0])
	if currIdx == -1 {
		currIdx = 0
		if p.currentTenant != "" {
			for i, t := range tenants {
				if t > p.currentTenant {
					currIdx = i
					break
				}
			}
		}
		p.currentTenant = tenants[currIdx]
		p.deficits[p.currentTenant] += p.quantum(p.currentTenant)
	}

	// Now check if currentTenant can serve its head request.
	// If not, advance to next tenant (replenishing its quantum) until a tenant can serve.
	n := len(tenants)
	maxIterations := n * 100 // Allow up to 100 full rounds of quantum accumulation
	for iter := 0; iter < maxIterations; iter++ {
		t := tenants[currIdx]
		head, ok := view.HeadRequest(t)
		if ok && head != nil {
			reqCost := p.cost(head)
			if p.deficits[t] >= reqCost {
				p.currentTenant = t
				return t
			}
		}

		// Tenant t cannot serve head request; advance to next tenant in cyclic order
		currIdx = (currIdx + 1) % n
		p.currentTenant = tenants[currIdx]
		p.deficits[p.currentTenant] += p.quantum(p.currentTenant)
	}

	// Fallback to first backlogged tenant
	p.currentTenant = tenants[0]
	return tenants[0]
}

func (p *DRR) OnDispatch(req RequestView) {
	p.mu.Lock()
	defer p.mu.Unlock()
	t := req.TenantID()
	reqCost := p.cost(req)
	p.deficits[t] -= reqCost
	if p.deficits[t] < 0 {
		p.deficits[t] = 0
	}
	p.currentTenant = t
}

func (p *DRR) OnArrival(req RequestView, queueWasEmpty bool, lastDepartedTenant string, otherBackloggedTenants []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if queueWasEmpty {
		p.deficits[req.TenantID()] = 0
	}
}

func (p *DRR) OnTokens(tenantID string, inputTokens, outputTokens int)                  {}
func (p *DRR) OnComplete(tenantID string, req RequestView, totalInput, totalOutput int) {}
