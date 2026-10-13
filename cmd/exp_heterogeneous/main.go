package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"fairlane/policy"
	"fairlane/sim"
)

type EnvironmentMetadata struct {
	GitCommit     string `json:"git_commit"`
	IsDirtyTree   bool   `json:"is_dirty_tree"`
	OS            string `json:"os"`
	CPU           string `json:"cpu"`
	Cores         int    `json:"cores"`
	LogicalCPUs   int    `json:"logical_cpus"`
	RAMGB         string `json:"ram_gb"`
	PowerPlan     string `json:"power_plan"`
	GoVersion     string `json:"go_version"`
	PythonVersion string `json:"python_version"`
}

func getEnvMetadata() EnvironmentMetadata {
	gitOut, _ := exec.Command("git", "rev-parse", "HEAD").Output()
	commit := strings.TrimSpace(string(gitOut))
	if commit == "" {
		commit = "uncommitted"
	}

	statusOut, _ := exec.Command("git", "status", "--porcelain").Output()
	isDirty := len(strings.TrimSpace(string(statusOut))) > 0

	return EnvironmentMetadata{
		GitCommit:     commit,
		IsDirtyTree:   isDirty,
		OS:            "Microsoft Windows 11 Pro 10.0.26300",
		CPU:           "AMD Ryzen 7 PRO 5850U with Radeon Graphics",
		Cores:         8,
		LogicalCPUs:   16,
		RAMGB:         "15.31 GB",
		PowerPlan:     "Balanced (381b4222-f694-41f0-9685-ff5bb260df2e)",
		GoVersion:     "go1.27.0 windows/amd64",
		PythonVersion: "Python 3.12.10",
	}
}

type SummaryStats struct {
	Count   int     `json:"count"`
	Mean    float64 `json:"mean"`
	Median  float64 `json:"median"`
	P95     float64 `json:"p95"`
	P99     float64 `json:"p99"`
	CI95Low float64 `json:"ci95_low"`
	CI95Hi  float64 `json:"ci95_hi"`
}

func bootstrapCI(values []float64, resamples int, alpha float64) (low, high float64) {
	n := len(values)
	if n == 0 {
		return 0, 0
	}
	if n == 1 {
		return values[0], values[0]
	}

	rng := rand.New(rand.NewSource(12345))
	means := make([]float64, resamples)

	for b := 0; b < resamples; b++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			idx := rng.Intn(n)
			sum += values[idx]
		}
		means[b] = sum / float64(n)
	}

	sort.Float64s(means)
	lowIdx := int(math.Floor(alpha / 2.0 * float64(resamples)))
	hiIdx := int(math.Ceil((1.0 - alpha/2.0) * float64(resamples)))
	if hiIdx >= resamples {
		hiIdx = resamples - 1
	}

	return means[lowIdx], means[hiIdx]
}

func computeSummary(values []float64) SummaryStats {
	n := len(values)
	if n == 0 {
		return SummaryStats{}
	}

	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)

	sum := 0.0
	for _, v := range sorted {
		sum += v
	}
	mean := sum / float64(n)

	getQuantile := func(q float64) float64 {
		if n == 1 {
			return sorted[0]
		}
		idx := q * float64(n-1)
		low := int(math.Floor(idx))
		high := int(math.Ceil(idx))
		if low == high {
			return sorted[low]
		}
		w := idx - float64(low)
		return sorted[low]*(1-w) + sorted[high]*w
	}

	ciLow, ciHi := bootstrapCI(values, 10000, 0.05)

	return SummaryStats{
		Count:   n,
		Mean:    mean,
		Median:  getQuantile(0.50),
		P95:     getQuantile(0.95),
		P99:     getQuantile(0.99),
		CI95Low: ciLow,
		CI95Hi:  ciHi,
	}
}

type Exp2PolicyResult struct {
	PolicyName           string       `json:"policy_name"`
	ChatRawTokenShare    SummaryStats `json:"chat_raw_token_share"`
	RAGRawTokenShare     SummaryStats `json:"rag_raw_token_share"`
	ChatWeightedShare    SummaryStats `json:"chat_weighted_token_share"`
	RAGWeightedShare     SummaryStats `json:"rag_weighted_token_share"`
	ChatWindowDispatches SummaryStats `json:"chat_window_dispatches"`
	RAGWindowDispatches  SummaryStats `json:"rag_window_dispatches"`
	ChatUnservedAtEnd    SummaryStats `json:"chat_unserved_at_end"`
	RAGUnservedAtEnd     SummaryStats `json:"rag_unserved_at_end"`
	ChatTTFTMeanMs       SummaryStats `json:"chat_ttft_mean_ms"`
	ChatTTFTP95Ms        SummaryStats `json:"chat_ttft_p95_ms"`
	RAGTTFTMeanMs        SummaryStats `json:"rag_ttft_mean_ms"`
	RAGTTFTP95Ms         SummaryStats `json:"rag_ttft_p95_ms"`
	MaxMinRatioWindow    SummaryStats `json:"max_min_ratio_window"`
	BothBackloggedSteps  SummaryStats `json:"both_backlogged_steps"`
	TotalWindowRawTokens SummaryStats `json:"total_window_raw_tokens"`
}

type Exp2SummaryFile struct {
	Experiment  string              `json:"experiment"`
	Environment EnvironmentMetadata `json:"environment"`
	Policies    []Exp2PolicyResult  `json:"policies"`
}

func main() {
	const (
		numSeeds        = 30
		seedStart int64 = 200
		warmupUs  int64 = 10000000 // 10.0s warmup
		horizonUs int64 = 60000000 // 60.0s fixed horizon
	)

	serverCfg := sim.ServerConfig{
		KVCapacityTokens: 16384,
		MaxBatchSize:     16,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	chatCfg := sim.TenantWorkloadConfig{
		TenantID:        "chat",
		RequestCount:    4000,  // 80s of arrivals (past 60s horizon)
		ArrivalProcess:  "poisson",
		MeanIntervalUs:  20000, // 50 req/s -> heavy backlog
		PromptLengthMin: 128,
		PromptLengthMax: 128,
		OutputLengthMin: 64,
		OutputLengthMax: 64,
	}

	ragCfg := sim.TenantWorkloadConfig{
		TenantID:        "rag",
		RequestCount:    800,    // 80s of arrivals (past 60s horizon)
		ArrivalProcess:  "poisson",
		MeanIntervalUs:  100000, // 10 req/s -> heavy backlog
		PromptLengthMin: 4000,
		PromptLengthMax: 4000,
		OutputLengthMin: 128,
		OutputLengthMax: 128,
	}

	policies := []struct {
		name string
		fn   func() policy.Policy
	}{
		{"fifo", func() policy.Policy { return policy.NewFIFO() }},
		{"round-robin-upstream", func() policy.Policy { return policy.NewRoundRobinWithAdvance(policy.RRAdvanceUpstream) }},
		{"round-robin-idealized", func() policy.Policy { return policy.NewRoundRobinWithAdvance(policy.RRAdvanceIdealized) }},
		{"vtc", func() policy.Policy { return policy.NewVTC(policy.DefaultVTCConfig()) }},
	}

	type SingleRunMetric struct {
		chatRawShare    float64
		ragRawShare     float64
		chatWeighted    float64
		ragWeighted     float64
		chatDisp        float64
		ragDisp         float64
		chatUnserved    float64
		ragUnserved     float64
		chatTTFTMean    float64
		chatTTFTP95     float64
		ragTTFTMean     float64
		ragTTFTP95      float64
		maxMinRatio     float64
		backloggedSteps float64
		totalRawTokens  float64
	}

	policyRuns := make(map[string][]SingleRunMetric)

	for s := int64(0); s < numSeeds; s++ {
		seed := seedStart + s
		gen := sim.NewWorkloadGenerator(seed)
		trace := gen.GenerateTrace([]sim.TenantWorkloadConfig{chatCfg, ragCfg})
		// CRITICAL: Sort trace by arrival time so chat and rag arrive concurrently
		sort.Slice(trace, func(i, j int) bool {
			return trace[i].ArrivalTime() < trace[j].ArrivalTime()
		})

		for _, p := range policies {
			pol := p.fn()
			qm := sim.NewTenantQueueManager()
			server := sim.NewServer(serverCfg)

			var nowUs int64 = 0
			traceIdx := 0

			var rawChat, rawRAG int
			var weightedChat, weightedRAG float64
			var winDispChat, winDispRAG int
			bothSteps := 0

			var chatTTFTs, ragTTFTs []int64

			for (nowUs < horizonUs) && (traceIdx < len(trace) || qm.TotalQueuedRequests() > 0 || server.RunningCount() > 0) {
				for traceIdx < len(trace) && trace[traceIdx].ArrivalTime() <= nowUs {
					qm.Enqueue(trace[traceIdx], pol, serverCfg.KVCapacityTokens)
					traceIdx++
				}
				qm.SetCurrentTime(nowUs)

				for server.RunningCount() < serverCfg.MaxBatchSize && qm.TotalQueuedRequests() > 0 {
					winner := pol.Pick(qm)
					if winner == "" {
						break
					}
					head, ok := qm.HeadRequest(winner)
					if !ok || head == nil {
						break
					}
					if server.CanAdmit(head) {
						req := qm.PopHead(winner)
						server.Admit(req, pol, nowUs)
						if nowUs >= warmupUs && nowUs < horizonUs {
							if winner == "chat" {
								winDispChat++
							} else {
								winDispRAG++
							}
						}
					} else {
						// Break on no fit (default)
						break
					}
				}

				bothBacklogged := qm.QueueDepth("chat") > 0 && qm.QueueDepth("rag") > 0

				if server.RunningCount() > 0 {
					res := server.Step(pol, nowUs)
					nowUs += res.StepDurationUs

					for _, comp := range res.Completed {
						ttft := comp.TTFT()
						if comp.TenantID() == "chat" {
							chatTTFTs = append(chatTTFTs, ttft)
						} else {
							ragTTFTs = append(ragTTFTs, ttft)
						}
					}

					// Measurement window: discard warm-up and condition on BOTH backlogged
					if nowUs >= warmupUs && nowUs < horizonUs && bothBacklogged {
						bothSteps++
						rawChat += res.TokensPerTenant["chat"]
						rawRAG += res.TokensPerTenant["rag"]
						weightedChat += res.WeightedTokensPerTenant["chat"]
						weightedRAG += res.WeightedTokensPerTenant["rag"]
					}
				} else {
					if qm.TotalQueuedRequests() > 0 {
						nowUs += 1000
					} else if traceIdx < len(trace) {
						nowUs = trace[traceIdx].ArrivalTime()
					} else {
						break
					}
				}
			}

			totalRaw := float64(rawChat + rawRAG)
			totalWeighted := weightedChat + weightedRAG

			var cRawShare, rRawShare, cWShare, rWShare float64
			mmRatio := 1.0
			if totalRaw > 0 {
				cRawShare = float64(rawChat) / totalRaw
				rRawShare = float64(rawRAG) / totalRaw
			}
			if totalWeighted > 0 {
				cWShare = weightedChat / totalWeighted
				rWShare = weightedRAG / totalWeighted
				if weightedChat > 0 && weightedRAG > 0 {
					if weightedChat > weightedRAG {
						mmRatio = weightedChat / weightedRAG
					} else {
						mmRatio = weightedRAG / weightedChat
					}
				}
			}

			cPerc := sim.CalculatePercentiles(chatTTFTs)
			rPerc := sim.CalculatePercentiles(ragTTFTs)

			unservedChat := float64(qm.QueueDepth("chat"))
			unservedRAG := float64(qm.QueueDepth("rag"))

			policyRuns[p.name] = append(policyRuns[p.name], SingleRunMetric{
				chatRawShare:    cRawShare,
				ragRawShare:     rRawShare,
				chatWeighted:    cWShare,
				ragWeighted:     rWShare,
				chatDisp:        float64(winDispChat),
				ragDisp:         float64(winDispRAG),
				chatUnserved:    unservedChat,
				ragUnserved:     unservedRAG,
				chatTTFTMean:    cPerc.Mean / 1000.0,
				chatTTFTP95:     cPerc.P95 / 1000.0,
				ragTTFTMean:     rPerc.Mean / 1000.0,
				ragTTFTP95:      rPerc.P95 / 1000.0,
				maxMinRatio:     mmRatio,
				backloggedSteps: float64(bothSteps),
				totalRawTokens:  totalRaw,
			})
		}
	}

	var results []Exp2PolicyResult

	extract := func(runs []SingleRunMetric, fn func(r SingleRunMetric) float64) []float64 {
		vals := make([]float64, len(runs))
		for i, r := range runs {
			vals[i] = fn(r)
		}
		return vals
	}

	for _, p := range policies {
		runs := policyRuns[p.name]
		res := Exp2PolicyResult{
			PolicyName:           p.name,
			ChatRawTokenShare:    computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.chatRawShare })),
			RAGRawTokenShare:     computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.ragRawShare })),
			ChatWeightedShare:    computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.chatWeighted })),
			RAGWeightedShare:     computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.ragWeighted })),
			ChatWindowDispatches: computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.chatDisp })),
			RAGWindowDispatches:  computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.ragDisp })),
			ChatUnservedAtEnd:    computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.chatUnserved })),
			RAGUnservedAtEnd:     computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.ragUnserved })),
			ChatTTFTMeanMs:       computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.chatTTFTMean })),
			ChatTTFTP95Ms:        computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.chatTTFTP95 })),
			RAGTTFTMeanMs:        computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.ragTTFTMean })),
			RAGTTFTP95Ms:         computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.ragTTFTP95 })),
			MaxMinRatioWindow:    computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.maxMinRatio })),
			BothBackloggedSteps:  computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.backloggedSteps })),
			TotalWindowRawTokens: computeSummary(extract(runs, func(r SingleRunMetric) float64 { return r.totalRawTokens })),
		}
		results = append(results, res)
	}

	summaryFile := Exp2SummaryFile{
		Experiment:  "exp_heterogeneous_steady_state",
		Environment: getEnvMetadata(),
		Policies:    results,
	}

	resultsDir := filepath.Join("experiments", "results")
	_ = os.MkdirAll(resultsDir, 0755)
	summaryPath := filepath.Join(resultsDir, "exp_heterogeneous_summary.json")

	bytes, err := json.MarshalIndent(summaryFile, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling JSON: %v\n", err)
		return
	}
	if err := os.WriteFile(summaryPath, bytes, 0644); err != nil {
		fmt.Printf("Error writing %s: %v\n", summaryPath, err)
		return
	}

	fmt.Println("\n=== EXPERIMENT 2: STEADY-STATE TOKEN SHARES (N=30 seeds, 95% CI) ===")
	fmt.Println("Window: [10.0s, 60.0s], conditioned on BOTH queues backlogged, sorted arrivals")
	fmt.Printf("| %-24s | %-24s | %-24s | %-16s |\n",
		"Policy", "Chat Wtd Share (w=1,2)", "Chat Raw Token Share", "Max-Min Ratio")
	fmt.Println("|--------------------------|--------------------------|--------------------------|------------------|")
	for _, r := range results {
		fmt.Printf("| %-24s | %5.1f%% [%4.1f%%, %4.1f%%]    | %5.1f%% [%4.1f%%, %4.1f%%]    | %5.2f [%4.2f,%4.2f] |\n",
			r.PolicyName,
			r.ChatWeightedShare.Mean*100, r.ChatWeightedShare.CI95Low*100, r.ChatWeightedShare.CI95Hi*100,
			r.ChatRawTokenShare.Mean*100, r.ChatRawTokenShare.CI95Low*100, r.ChatRawTokenShare.CI95Hi*100,
			r.MaxMinRatioWindow.Mean, r.MaxMinRatioWindow.CI95Low, r.MaxMinRatioWindow.CI95Hi)
	}

	fmt.Println("\nDispatches in Window vs Unserved Requests at Horizon (N=30 seeds, 95% CI):")
	fmt.Printf("| %-24s | %-18s | %-18s | %-18s | %-18s |\n",
		"Policy", "Win Dispatches Chat", "Win Dispatches RAG", "Unserved Chat", "Unserved RAG")
	fmt.Println("|--------------------------|--------------------|--------------------|--------------------|--------------------|")
	for _, r := range results {
		fmt.Printf("| %-24s | %6.0f [%4.0f, %4.0f]  | %6.0f [%4.0f, %4.0f]  | %6.0f [%4.0f, %4.0f]  | %6.0f [%4.0f, %4.0f]  |\n",
			r.PolicyName,
			r.ChatWindowDispatches.Mean, r.ChatWindowDispatches.CI95Low, r.ChatWindowDispatches.CI95Hi,
			r.RAGWindowDispatches.Mean, r.RAGWindowDispatches.CI95Low, r.RAGWindowDispatches.CI95Hi,
			r.ChatUnservedAtEnd.Mean, r.ChatUnservedAtEnd.CI95Low, r.ChatUnservedAtEnd.CI95Hi,
			r.RAGUnservedAtEnd.Mean, r.RAGUnservedAtEnd.CI95Low, r.RAGUnservedAtEnd.CI95Hi)
	}

	fmt.Println("\nComparison with Hand-Calculated Predictions:")
	fmt.Printf("| %-24s | %-16s | %-18s | %-16s | %-18s |\n",
		"Policy", "Hand-Calc Wtd", "Measured Wtd CI", "Hand-Calc Raw", "Measured Raw CI")
	fmt.Println("|--------------------------|------------------|--------------------|------------------|--------------------|")
	for _, r := range results {
		var predWtd, predRaw string
		switch r.PolicyName {
		case "fifo":
			predWtd = "~22.8% (arr mix)"
			predRaw = "~18.9% (arr mix)"
		case "round-robin-idealized":
			predWtd = "~5.7% (1:1 req)"
			predRaw = "~4.4% (1:1 req)"
		case "round-robin-upstream":
			predWtd = "~30%-35% (bypass)"
			predRaw = "~25%-30% (bypass)"
		case "vtc":
			predWtd = "~50.0% (token-fair)"
			predRaw = "~43.6% (token-fair)"
		}
		fmt.Printf("| %-24s | %-16s | %5.1f%% [%4.1f-%4.1f] | %-16s | %5.1f%% [%4.1f-%4.1f] |\n",
			r.PolicyName, predWtd,
			r.ChatWeightedShare.Mean*100, r.ChatWeightedShare.CI95Low*100, r.ChatWeightedShare.CI95Hi*100,
			predRaw,
			r.ChatRawTokenShare.Mean*100, r.ChatRawTokenShare.CI95Low*100, r.ChatRawTokenShare.CI95Hi*100)
	}

	fmt.Println("\nTTFT Performance (Mean / P95):")
	fmt.Printf("| %-24s | %-26s | %-26s |\n",
		"Policy", "Chat TTFT Mean ms (95% CI)", "RAG TTFT Mean ms (95% CI)")
	fmt.Println("|--------------------------|----------------------------|----------------------------|")
	for _, r := range results {
		fmt.Printf("| %-24s | %6.1f [%5.1f, %5.1f] ms   | %6.1f [%5.1f, %5.1f] ms   |\n",
			r.PolicyName,
			r.ChatTTFTMeanMs.Mean, r.ChatTTFTMeanMs.CI95Low, r.ChatTTFTMeanMs.CI95Hi,
			r.RAGTTFTMeanMs.Mean, r.RAGTTFTMeanMs.CI95Low, r.RAGTTFTMeanMs.CI95Hi)
	}

	fmt.Printf("\nSaved summary to: %s\n", summaryPath)
}
