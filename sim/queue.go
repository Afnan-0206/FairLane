package sim

import (
	"sort"

	"fairlane/policy"
)

// TenantQueueManager owns the multi-tenant FIFO waiting queues and implements policy.SchedulerView.
type TenantQueueManager struct {
	queues             map[string][]*Request
	lastDepartedTenant string
	currentTimeUs      int64
	neverFitsRejected  int
}

func NewTenantQueueManager() *TenantQueueManager {
	return &TenantQueueManager{
		queues: make(map[string][]*Request),
	}
}

func (m *TenantQueueManager) SetCurrentTime(nowUs int64) {
	m.currentTimeUs = nowUs
}

func (m *TenantQueueManager) CurrentTime() int64 {
	return m.currentTimeUs
}

func (m *TenantQueueManager) NeverFitsRejected() int {
	return m.neverFitsRejected
}

func (m *TenantQueueManager) LastDepartedTenant() string {
	return m.lastDepartedTenant
}

// BackloggedTenants returns the slice of tenant IDs with non-empty waiting queues.
// INVARIANT: Lexicographically sorted to guarantee determinism across all platforms.
func (m *TenantQueueManager) BackloggedTenants() []string {
	var res []string
	for t, q := range m.queues {
		if len(q) > 0 {
			res = append(res, t)
		}
	}
	sort.Strings(res)
	return res
}

// HeadRequest returns the request at the head of the tenant's queue without popping.
func (m *TenantQueueManager) HeadRequest(tenantID string) (policy.RequestView, bool) {
	q, ok := m.queues[tenantID]
	if !ok || len(q) == 0 {
		return nil, false
	}
	return q[0], true
}

// Enqueue adds a request to the tenant's queue.
// If inputTokens + 1 > kvCapacity, the request can NEVER fit in the server.
// It is rejected immediately per Amendment 4.
func (m *TenantQueueManager) Enqueue(req *Request, pol policy.Policy, kvCapacity int) bool {
	if req.inputTokens+1 > kvCapacity {
		m.neverFitsRejected++
		return false
	}

	tenant := req.tenantID
	q := m.queues[tenant]
	wasEmpty := (len(q) == 0)

	// Identify other active backlogged tenants prior to this enqueue
	var otherBacklogged []string
	for t, tq := range m.queues {
		if t != tenant && len(tq) > 0 {
			otherBacklogged = append(otherBacklogged, t)
		}
	}
	sort.Strings(otherBacklogged)

	m.queues[tenant] = append(q, req)

	// Notify policy of arrival with accurate transition state
	pol.OnArrival(req, wasEmpty, m.lastDepartedTenant, otherBacklogged)
	return true
}

// PopHead removes and returns the head request for the winning tenant.
// If the tenant's queue becomes empty, it marks the tenant as departed (left Q).
func (m *TenantQueueManager) PopHead(tenantID string) *Request {
	q, ok := m.queues[tenantID]
	if !ok || len(q) == 0 {
		return nil
	}

	req := q[0]
	m.queues[tenantID] = q[1:]

	// "Left Q" transition: tenant has cleared all waiting requests
	if len(m.queues[tenantID]) == 0 {
		m.lastDepartedTenant = tenantID
	}

	return req
}

// TotalQueuedRequests returns the aggregate number of waiting requests across all tenants.
func (m *TenantQueueManager) TotalQueuedRequests() int {
	total := 0
	for _, q := range m.queues {
		total += len(q)
	}
	return total
}

// QueueDepth returns the number of waiting requests for a specific tenant.
func (m *TenantQueueManager) QueueDepth(tenantID string) int {
	return len(m.queues[tenantID])
}

// Ensure TenantQueueManager satisfies policy.SchedulerView
var _ policy.SchedulerView = (*TenantQueueManager)(nil)
