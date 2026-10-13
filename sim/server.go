package sim

import (
	"fairlane/policy"
)

// ServerConfig configures the continuous batching model server.
type ServerConfig struct {
	KVCapacityTokens int   // Maximum total tokens allowed in KV cache
	MaxBatchSize     int   // Maximum concurrent requests in flight
	BaseStepTimeUs   int64 // Base overhead per execution step (microseconds)
	PerTokenCostUs   int64 // Incremental execution cost per active token in step (microseconds)
}

// DefaultServerConfig provides realistic baseline parameters for a continuous batching model.
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		KVCapacityTokens: 4096,
		MaxBatchSize:     32,
		BaseStepTimeUs:   5000, // 5 ms base step time
		PerTokenCostUs:   5,    // 5 µs per token in batch
	}
}

// RunningRequest tracks a request actively executing on the server.
type RunningRequest struct {
	req         *Request
	isPrefill   bool // true if currently performing initial prompt prefill
	tokensInKV  int  // inputTokens + generatedCount
}

// Server simulates an LLM inference engine with continuous batching.
type Server struct {
	cfg        ServerConfig
	running    []*RunningRequest
	usedKV     int
	currentTime int64
}

func NewServer(cfg ServerConfig) *Server {
	return &Server{
		cfg: cfg,
	}
}

func (s *Server) RunningCount() int {
	return len(s.running)
}

func (s *Server) UsedKVTokens() int {
	return s.usedKV
}

func (s *Server) FreeKVTokens() int {
	return s.cfg.KVCapacityTokens - s.usedKV
}

// CanAdmit checks whether a candidate request can fit into the current batch and KV cache.
// A request requires inputTokens + 1 (first decode token) of KV space to be admitted.
func (s *Server) CanAdmit(req policy.RequestView) bool {
	if len(s.running) >= s.cfg.MaxBatchSize {
		return false
	}
	needed := req.InputTokens() + 1
	return s.usedKV+needed <= s.cfg.KVCapacityTokens
}

// Admit places a request into the running batch.
func (s *Server) Admit(req *Request, pol policy.Policy, nowUs int64) {
	req.dispatchedUs = nowUs
	running := &RunningRequest{
		req:        req,
		isPrefill:  true,
		tokensInKV: req.inputTokens,
	}
	s.running = append(s.running, running)
	s.usedKV += req.inputTokens

	// Notify policy of admission
	pol.OnDispatch(req)
}

// StepResult summarizes the execution of one continuous batching step.
type StepResult struct {
	StepDurationUs          int64
	TokensInStep            int
	TokensPerTenant         map[string]int
	WeightedTokensPerTenant map[string]float64
	Completed               []*Request
}

// Step advances execution by one continuous batching step.
// For prefill requests: processes the full prompt, transitions to decode, marks TTFT.
// For decode requests: generates 1 output token, checks for completion.
func (s *Server) Step(pol policy.Policy, nowUs int64) StepResult {
	if len(s.running) == 0 {
		return StepResult{}
	}

	tokensInStep := 0
	tokensPerTenant := make(map[string]int)
	weightedTokensPerTenant := make(map[string]float64)
	var completed []*Request
	var remaining []*RunningRequest

	// First pass: calculate step workload and step duration
	for _, r := range s.running {
		if r.isPrefill {
			tokensInStep += r.req.inputTokens
			tokensPerTenant[r.req.tenantID] += r.req.inputTokens
			weightedTokensPerTenant[r.req.tenantID] += float64(r.req.inputTokens) * 1.0
		} else {
			tokensInStep += 1
			tokensPerTenant[r.req.tenantID] += 1
			weightedTokensPerTenant[r.req.tenantID] += 2.0
		}
	}

	stepDuration := s.cfg.BaseStepTimeUs + s.cfg.PerTokenCostUs*int64(tokensInStep)
	stepFinishUs := nowUs + stepDuration

	// Second pass: advance requests and notify policy
	for _, r := range s.running {
		r.req.stepLatencies = append(r.req.stepLatencies, stepDuration)

		if r.isPrefill {
			// Prefill complete: transition to decode
			r.isPrefill = false
			r.req.firstTokenUs = stepFinishUs
			r.req.generatedCount = 1
			r.tokensInKV += 1
			s.usedKV += 1

			// Incremental charging: notify generated token
			pol.OnTokens(r.req.tenantID, 0, 1)

			if r.req.generatedCount >= r.req.actualOut {
				// Request finished on first step (e.g. 1-token output)
				r.req.completedUs = stepFinishUs
				s.usedKV -= r.tokensInKV
				pol.OnComplete(r.req.tenantID, r.req, r.req.inputTokens, r.req.generatedCount)
				completed = append(completed, r.req)
			} else {
				remaining = append(remaining, r)
			}
		} else {
			// Decode step: generate 1 token
			r.req.generatedCount++
			r.tokensInKV += 1
			s.usedKV += 1

			pol.OnTokens(r.req.tenantID, 0, 1)

			if r.req.generatedCount >= r.req.actualOut {
				// Request finished
				r.req.completedUs = stepFinishUs
				s.usedKV -= r.tokensInKV
				pol.OnComplete(r.req.tenantID, r.req, r.req.inputTokens, r.req.generatedCount)
				completed = append(completed, r.req)
			} else {
				remaining = append(remaining, r)
			}
		}
	}

	s.running = remaining
	return StepResult{
		StepDurationUs:          stepDuration,
		TokensInStep:            tokensInStep,
		TokensPerTenant:         tokensPerTenant,
		WeightedTokensPerTenant: weightedTokensPerTenant,
		Completed:               completed,
	}
}
