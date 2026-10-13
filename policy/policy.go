package policy

// RequestView exposes only policy-visible request metadata.
// CRITICAL INVARIANT: The true output length is STRICTLY ABSENT from RequestView.
// Schedulers can only observe the tenant, input token count, arrival time,
// and an optional predicted output token count.
type RequestView interface {
	ID() string
	TenantID() string
	InputTokens() int
	PredictedOutputTokens() int // 0 if no prediction available
	ArrivalTime() int64         // Virtual time in microseconds
}

// SchedulerView represents the read-only state of tenant queues exposed to the fairness policy.
// The framework (simulator or proxy engine) owns the queues.
type SchedulerView interface {
	// BackloggedTenants returns the slice of tenant IDs that have at least one waiting request.
	// INVARIANT: The returned slice MUST be lexicographically sorted to prevent non-deterministic map iteration.
	BackloggedTenants() []string

	// HeadRequest returns the request at the head of the tenant's FIFO queue, if any.
	HeadRequest(tenantID string) (RequestView, bool)

	// CurrentTime returns the current virtual time in microseconds.
	CurrentTime() int64
}

// LiftMode defines how counter lift handles the empty-queue case.
type LiftMode int

const (
	// LiftModePaper follows arXiv:2401.00588 Algorithm 2 lines 9-11:
	// When Q is empty, c_u is lifted to max(c_u, c_l) where l is the last tenant that departed Q.
	LiftModePaper LiftMode = iota

	// LiftModeArtifact follows the reference implementation (slora/server/router/vtc_req_queue.py):
	// When Q is empty (len(cnts) == 0), no lift occurs; c_u is unchanged.
	LiftModeArtifact
)

func (m LiftMode) String() string {
	switch m {
	case LiftModePaper:
		return "paper"
	case LiftModeArtifact:
		return "artifact"
	default:
		return "unknown"
	}
}

// LiftTiming defines when VTC counter lift is evaluated.
type LiftTiming int

const (
	// LiftTimingArrival evaluates counter lift at the instant a request arrives to an empty queue.
	LiftTimingArrival LiftTiming = iota

	// LiftTimingPick evaluates counter lift lazily at dispatch time when selecting among backlogged tenants.
	LiftTimingPick
)

func (t LiftTiming) String() string {
	switch t {
	case LiftTimingArrival:
		return "arrival"
	case LiftTimingPick:
		return "pick"
	default:
		return "unknown"
	}
}

// ChargingModel defines when token costs are charged to tenant counters.
type ChargingModel int

const (
	// ChargingIncremental charges input tokens at dispatch and output tokens step-by-step as generated.
	ChargingIncremental ChargingModel = iota

	// ChargingCompletion charges the full request token cost (input + output) upon completion.
	ChargingCompletion
)

func (c ChargingModel) String() string {
	switch c {
	case ChargingIncremental:
		return "incremental"
	case ChargingCompletion:
		return "completion"
	default:
		return "unknown"
	}
}

// Policy governs inter-tenant flow selection (Tier 2 in llm-d dispatch hierarchy).
// It answers: "Which backlogged tenant gets the next dispatch opportunity?"
type Policy interface {
	Name() string

	// Pick inspects backlogged tenants in the scheduler view and returns the winning tenant ID.
	// Returns "" if no backlogged tenant is eligible.
	// INVARIANT: Ties MUST be broken by lexicographically lowest tenant ID.
	Pick(view SchedulerView) (tenantID string)

	// OnArrival notifies the policy that a request has arrived for a tenant.
	// queueWasEmpty is true if the tenant's queue was empty before this arrival.
	// lastDepartedTenant is the ID of the tenant whose queue most recently became empty.
	// otherBackloggedTenants contains the IDs of all other tenants currently having queued requests.
	OnArrival(req RequestView, queueWasEmpty bool, lastDepartedTenant string, otherBackloggedTenants []string)

	// OnDispatch notifies the policy that a request from a tenant was admitted into a running batch.
	OnDispatch(req RequestView)

	// OnTokens notifies the policy that new tokens were processed for a tenant.
	// Used in incremental charging mode.
	OnTokens(tenantID string, inputTokens, outputTokens int)

	// OnComplete notifies the policy that a request has finished processing.
	OnComplete(tenantID string, req RequestView, totalInput, totalOutput int)
}
