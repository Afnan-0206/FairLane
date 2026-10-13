package sim

import (
	"sort"

	"fairlane/policy"
)

// AdmissionMode defines whether non-fitting head requests block batch formation or are skipped.
type AdmissionMode int

const (
	// AdmissionModeBreak halts batch formation when the winner's head request cannot fit into the available KV cache.
	// This preserves strict order and prevents large-request starvation (Default).
	AdmissionModeBreak AdmissionMode = iota

	// AdmissionModeSkip bypasses a winner whose head request cannot fit, evaluating other backlogged tenants.
	// Can cause large-request starvation under continuous small-request load.
	AdmissionModeSkip
)

func (m AdmissionMode) String() string {
	switch m {
	case AdmissionModeBreak:
		return "break"
	case AdmissionModeSkip:
		return "skip"
	default:
		return "unknown"
	}
}

// Simulator drives the discrete-event execution of workload traces through a fairness policy.
type Simulator struct {
	server       *Server
	queueManager *TenantQueueManager
	policy       policy.Policy

	admissionMode AdmissionMode

	slidingWindowTokens int
	windowAllocations   []map[string]float64 // historical token allocations per token window

	timeWindowUs         int64
	timeWindowAllocs     []map[string]float64 // historical token allocations per time window
	timeWindowDemands    []map[string]float64 // active tenant demands per time window

	// Tracking metrics
	idleWhileQueuedUs int64
	completedRequests []*Request
	submittedCount    map[string]int
}

func NewSimulator(serverCfg ServerConfig, pol policy.Policy, slidingWindowTokens int) *Simulator {
	if slidingWindowTokens <= 0 {
		slidingWindowTokens = 1000 // default sliding window of 1000 tokens
	}
	return &Simulator{
		server:              NewServer(serverCfg),
		queueManager:        NewTenantQueueManager(),
		policy:              pol,
		admissionMode:       AdmissionModeBreak, // Default is break-on-no-fit
		slidingWindowTokens: slidingWindowTokens,
		timeWindowUs:        500000, // 500ms time window
		submittedCount:      make(map[string]int),
	}
}

func (s *Simulator) SetAdmissionMode(mode AdmissionMode) {
	s.admissionMode = mode
}

func (s *Simulator) AdmissionMode() AdmissionMode {
	return s.admissionMode
}

func (s *Simulator) SetTimeWindowUs(winUs int64) {
	if winUs > 0 {
		s.timeWindowUs = winUs
	}
}

// filteredSchedulerView filters out temporarily skipped tenants for AdmissionModeSkip.
type filteredSchedulerView struct {
	base    policy.SchedulerView
	skipped map[string]bool
}

func newFilteredSchedulerView(base policy.SchedulerView, skipped map[string]bool) *filteredSchedulerView {
	return &filteredSchedulerView{base: base, skipped: skipped}
}

func (v *filteredSchedulerView) BackloggedTenants() []string {
	var res []string
	for _, t := range v.base.BackloggedTenants() {
		if !v.skipped[t] {
			res = append(res, t)
		}
	}
	return res
}

func (v *filteredSchedulerView) HeadRequest(tenantID string) (policy.RequestView, bool) {
	if v.skipped[tenantID] {
		return nil, false
	}
	return v.base.HeadRequest(tenantID)
}

func (v *filteredSchedulerView) CurrentTime() int64 {
	return v.base.CurrentTime()
}

// Run executes the simulation until all requests in the trace complete.
func (s *Simulator) Run(trace []*Request) SimulationMetrics {
	// Sort trace deterministically by arrival time, tie-breaking by ID
	sortedTrace := make([]*Request, len(trace))
	copy(sortedTrace, trace)
	sort.Slice(sortedTrace, func(i, j int) bool {
		if sortedTrace[i].arrivalUs == sortedTrace[j].arrivalUs {
			return sortedTrace[i].id < sortedTrace[j].id
		}
		return sortedTrace[i].arrivalUs < sortedTrace[j].arrivalUs
	})

	for _, r := range sortedTrace {
		s.submittedCount[r.tenantID]++
	}

	var nowUs int64 = 0
	traceIdx := 0
	totalRequests := len(sortedTrace)

	// Sliding window tracking
	currentWindowTokens := 0
	currentWindowAlloc := make(map[string]float64)

	var lastTimeWindowUs int64 = 0
	currentTimeWindowAlloc := make(map[string]float64)

	for traceIdx < totalRequests || s.queueManager.TotalQueuedRequests() > 0 || s.server.RunningCount() > 0 {
		// 1. Enqueue all requests that have arrived up to nowUs
		for traceIdx < totalRequests && sortedTrace[traceIdx].arrivalUs <= nowUs {
			req := sortedTrace[traceIdx]
			s.queueManager.Enqueue(req, s.policy, s.server.cfg.KVCapacityTokens)
			traceIdx++
		}

		s.queueManager.SetCurrentTime(nowUs)

		// 2. Batch formation: Admit eligible requests using policy.Pick
		if s.admissionMode == AdmissionModeBreak {
			for s.server.RunningCount() < s.server.cfg.MaxBatchSize && s.queueManager.TotalQueuedRequests() > 0 {
				winnerTenant := s.policy.Pick(s.queueManager)
				if winnerTenant == "" {
					break
				}

				headReq, ok := s.queueManager.HeadRequest(winnerTenant)
				if !ok || headReq == nil {
					break
				}

				if s.server.CanAdmit(headReq) {
					req := s.queueManager.PopHead(winnerTenant)
					s.server.Admit(req, s.policy, nowUs)
				} else {
					// BREAK-ON-NO-FIT (Default):
					// Strict work-conserving order: if the head request of the selected tenant
					// cannot fit into available KV cache right now, break out of batch formation.
					break
				}
			}
		} else {
			// AdmissionModeSkip:
			// If a winning tenant's head request cannot fit, temporarily skip it in this batch formation round
			// and pick from the remaining backlogged tenants.
			skipped := make(map[string]bool)
			for s.server.RunningCount() < s.server.cfg.MaxBatchSize {
				filteredView := newFilteredSchedulerView(s.queueManager, skipped)
				if len(filteredView.BackloggedTenants()) == 0 {
					break
				}

				winnerTenant := s.policy.Pick(filteredView)
				if winnerTenant == "" {
					break
				}

				headReq, ok := s.queueManager.HeadRequest(winnerTenant)
				if !ok || headReq == nil {
					skipped[winnerTenant] = true
					continue
				}

				if s.server.CanAdmit(headReq) {
					req := s.queueManager.PopHead(winnerTenant)
					s.server.Admit(req, s.policy, nowUs)
				} else {
					// Cannot fit: skip this tenant for this admission round
					skipped[winnerTenant] = true
				}
			}
		}

		// 3. Execution step or idle advance
		if s.server.RunningCount() > 0 {
			// Continuous batching step
			res := s.server.Step(s.policy, nowUs)
			nowUs += res.StepDurationUs

			// Track window token allocations for Jain's fairness index
			for _, req := range res.Completed {
				s.completedRequests = append(s.completedRequests, req)
			}

			tokensThisStep := res.TokensInStep
			currentWindowTokens += tokensThisStep

			// Distribute step tokens to active running tenants for window accounting
			for _, r := range s.server.running {
				cost := 1.0
				if r.isPrefill {
					cost = float64(r.req.inputTokens)
				}
				currentWindowAlloc[r.req.tenantID] += cost
				currentTimeWindowAlloc[r.req.tenantID] += cost
			}

			if currentWindowTokens >= s.slidingWindowTokens {
				// Snapshot token window
				snap := make(map[string]float64, len(currentWindowAlloc))
				for k, v := range currentWindowAlloc {
					snap[k] = v
				}
				s.windowAllocations = append(s.windowAllocations, snap)
				currentWindowAlloc = make(map[string]float64)
				currentWindowTokens = 0
			}

			if nowUs-lastTimeWindowUs >= s.timeWindowUs {
				// Snapshot time window
				snapTime := make(map[string]float64, len(currentTimeWindowAlloc))
				for k, v := range currentTimeWindowAlloc {
					snapTime[k] = v
				}

				snapDemand := make(map[string]float64)
				for _, tid := range s.queueManager.BackloggedTenants() {
					snapDemand[tid] = float64(s.queueManager.QueueDepth(tid) * 100)
				}
				for _, r := range s.server.running {
					snapDemand[r.req.tenantID] += float64(r.req.inputTokens + r.req.actualOut)
				}

				s.timeWindowAllocs = append(s.timeWindowAllocs, snapTime)
				s.timeWindowDemands = append(s.timeWindowDemands, snapDemand)

				currentTimeWindowAlloc = make(map[string]float64)
				lastTimeWindowUs = nowUs
			}
		} else {
			// Server is idle (running == 0)
			queuedCount := s.queueManager.TotalQueuedRequests()
			if queuedCount > 0 {
				// IDLE-WHILE-QUEUED VIOLATION TRACKING:
				// If server is idle while requests are waiting, record the anomaly
				s.idleWhileQueuedUs += 1000 // 1ms idle tick
				nowUs += 1000
			} else if traceIdx < totalRequests {
				// Advance time to next arrival
				nowUs = sortedTrace[traceIdx].arrivalUs
			} else {
				break
			}
		}
	}

	return s.calculateMetrics(nowUs)
}

func (s *Simulator) calculateMetrics(totalTimeUs int64) SimulationMetrics {
	perTenant := make(map[string]*TenantMetrics)

	totalTokens := 0
	tenantTokens := make(map[string]float64)

	for _, req := range s.completedRequests {
		t, ok := perTenant[req.tenantID]
		if !ok {
			t = &TenantMetrics{
				TenantID:          req.tenantID,
				RequestsSubmitted: s.submittedCount[req.tenantID],
			}
			perTenant[req.tenantID] = t
		}

		t.RequestsCompleted++
		t.InputTokensServed += req.inputTokens
		t.OutputTokensServed += req.generatedCount
		t.TotalTokensServed += req.inputTokens + req.generatedCount

		totalTokens += req.inputTokens + req.generatedCount
		tenantTokens[req.tenantID] += float64(req.inputTokens + req.generatedCount)
	}

	// Calculate latency percentiles per tenant
	for tid, tm := range perTenant {
		var ttftList, itlList, latList []int64
		for _, req := range s.completedRequests {
			if req.tenantID == tid {
				ttftList = append(ttftList, req.TTFT())
				latList = append(latList, req.TotalLatency())
				if len(req.stepLatencies) > 1 {
					var sumItl int64
					for _, l := range req.stepLatencies[1:] {
						sumItl += l
					}
					itlList = append(itlList, sumItl/int64(len(req.stepLatencies)-1))
				}
			}
		}

		tm.TTFTUs = CalculatePercentiles(ttftList)
		tm.TotalLatencyUs = CalculatePercentiles(latList)
		tm.ITLUs = CalculatePercentiles(itlList)
	}

	// Sliding window Jain's index across windows
	var jainSum float64
	var maxMinSum float64
	validWindows := 0

	for _, win := range s.windowAllocations {
		if len(win) > 1 {
			var allocs []float64
			for _, v := range win {
				allocs = append(allocs, v)
			}
			jainSum += JainsIndex(allocs)
			maxMinSum += MaxMinRatio(allocs)
			validWindows++
		}
	}

	meanJain := 1.0
	meanMaxMin := 1.0
	if validWindows > 0 {
		meanJain = jainSum / float64(validWindows)
		meanMaxMin = maxMinSum / float64(validWindows)
	} else if len(tenantTokens) > 1 {
		var allocs []float64
		for _, v := range tenantTokens {
			allocs = append(allocs, v)
		}
		meanJain = JainsIndex(allocs)
		meanMaxMin = MaxMinRatio(allocs)
	}

	// Time-window demand-capped Jain index
	var jainDemandCappedSum float64
	validTimeWindows := 0
	for i, win := range s.timeWindowAllocs {
		if len(win) > 1 {
			demands := s.timeWindowDemands[i]
			var servedList, demandList []float64
			for tid, served := range win {
				servedList = append(servedList, served)
				demandList = append(demandList, demands[tid])
			}
			jainDemandCappedSum += JainsDemandCapped(servedList, demandList)
			validTimeWindows++
		}
	}

	meanJainDemandCapped := meanJain
	if validTimeWindows > 0 {
		meanJainDemandCapped = jainDemandCappedSum / float64(validTimeWindows)
	}

	totalSec := float64(totalTimeUs) / 1e6
	goodputTok := 0.0
	goodputReq := 0.0
	if totalSec > 0 {
		goodputTok = float64(totalTokens) / totalSec
		goodputReq = float64(len(s.completedRequests)) / totalSec
	}

	return SimulationMetrics{
		PolicyName:             s.policy.Name(),
		TotalVirtualTimeUs:     totalTimeUs,
		IdleWhileQueuedUs:      s.idleWhileQueuedUs,
		NeverFitsRejected:      s.queueManager.NeverFitsRejected(),
		TotalCompletedReqs:     len(s.completedRequests),
		TotalTokensServed:      totalTokens,
		GoodputTokensPerSec:    goodputTok,
		GoodputReqsPerSec:      goodputReq,
		JainsIndexSliding:      meanJain,
		MaxMinRatioSliding:     meanMaxMin,
		SlidingWindowTokens:    s.slidingWindowTokens,
		TimeWindowUs:           s.timeWindowUs,
		JainsIndexDemandCapped: meanJainDemandCapped,
		PerTenant:              perTenant,
	}
}
