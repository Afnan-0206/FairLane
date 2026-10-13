package policy

// FIFO implements global First-Come-First-Served flow selection.
// It maps directly to llm-d-router's global-strict-fairness-policy combined with fcfs-ordering-policy:
// It peeks at the head of every backlogged tenant's queue and picks the one with the earliest arrival timestamp.
type FIFO struct{}

func NewFIFO() *FIFO {
	return &FIFO{}
}

func (p *FIFO) Name() string {
	return "fifo"
}

func (p *FIFO) Pick(view SchedulerView) string {
	tenants := view.BackloggedTenants()
	if len(tenants) == 0 {
		return ""
	}

	var bestTenant string
	var earliestArrival int64 = -1

	for _, t := range tenants {
		head, ok := view.HeadRequest(t)
		if !ok || head == nil {
			continue
		}

		arr := head.ArrivalTime()
		if earliestArrival == -1 || arr < earliestArrival {
			earliestArrival = arr
			bestTenant = t
		} else if arr == earliestArrival {
			// Deterministic tie-break: lexicographically lowest tenant ID
			if bestTenant == "" || t < bestTenant {
				bestTenant = t
			}
		}
	}

	return bestTenant
}

func (p *FIFO) OnArrival(req RequestView, queueWasEmpty bool, lastDepartedTenant string, otherBackloggedTenants []string) {
}
func (p *FIFO) OnDispatch(req RequestView)                                               {}
func (p *FIFO) OnTokens(tenantID string, inputTokens, outputTokens int)                  {}
func (p *FIFO) OnComplete(tenantID string, req RequestView, totalInput, totalOutput int) {}
