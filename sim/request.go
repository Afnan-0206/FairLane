package sim

import "fairlane/policy"

// Request represents a simulated inference request with ground-truth generation lengths.
// It implements policy.RequestView for the scheduler, exposing only input tokens,
// arrival time, and optional predicted output tokens (NEVER actual output tokens).
type Request struct {
	id          string
	tenantID    string
	inputTokens int
	actualOut   int
	predOut     int
	priority    int // Priority band (0 = High/Default, 1 = Low)
	arrivalUs   int64

	// Simulation tracking state
	dispatchedUs   int64
	firstTokenUs   int64
	completedUs    int64
	generatedCount int
	stepLatencies  []int64 // Microseconds for each decode step
}

func NewRequest(id, tenantID string, inputTokens, actualOut, predOut int, arrivalUs int64) *Request {
	return &Request{
		id:          id,
		tenantID:    tenantID,
		inputTokens: inputTokens,
		actualOut:   actualOut,
		predOut:     predOut,
		priority:    0,
		arrivalUs:   arrivalUs,
	}
}

// WithPriority sets the priority band for the request (fluent builder).
func (r *Request) WithPriority(p int) *Request {
	r.priority = p
	return r
}

// SetPredictedOutput sets the predicted output tokens.
func (r *Request) SetPredictedOutput(p int) {
	r.predOut = p
}

// policy.RequestView implementation:
func (r *Request) ID() string                    { return r.id }
func (r *Request) TenantID() string              { return r.tenantID }
func (r *Request) InputTokens() int              { return r.inputTokens }
func (r *Request) PredictedOutputTokens() int     { return r.predOut }
func (r *Request) ArrivalTime() int64            { return r.arrivalUs }
func (r *Request) Priority() int                 { return r.priority }

// Simulation-internal getters (never accessed by policy):
func (r *Request) ActualOutputTokens() int { return r.actualOut }
func (r *Request) DispatchedTime() int64   { return r.dispatchedUs }
func (r *Request) FirstTokenTime() int64   { return r.firstTokenUs }
func (r *Request) CompletedTime() int64    { return r.completedUs }
func (r *Request) TTFT() int64             { return r.firstTokenUs - r.arrivalUs }
func (r *Request) TotalLatency() int64     { return r.completedUs - r.arrivalUs }

func (r *Request) MeanITL() float64 {
	if len(r.stepLatencies) <= 1 {
		return 0
	}
	var total int64
	for _, l := range r.stepLatencies[1:] {
		total += l
	}
	return float64(total) / float64(len(r.stepLatencies)-1)
}

// Ensure Request satisfies policy.RequestView at compile time
var _ policy.RequestView = (*Request)(nil)
