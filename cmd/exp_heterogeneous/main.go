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
	ChatRequestShare     SummaryStats `json:"chat_request_share"`
	RAGRequestShare      SummaryStats `json:"rag_request_share"`
	ChatTTFTMeanMs       SummaryStats `json:"chat_ttft_mean_ms"`
	ChatTTFTP95Ms        SummaryStats `json:"chat_ttft_p95_ms"`
	RAGTTFTMeanMs        SummaryStats `json:"rag_ttft_mean_ms"`
	RAGTTFTP95Ms         SummaryStats `json:"rag_ttft_p95_ms"`
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
		RequestCount:    2500,
		ArrivalProcess:  "poisson",
		MeanIntervalUs:  20000, // 50 req/s -> heavy backlog
		PromptLengthMin: 128,
		PromptLengthMax: 128,
		OutputLengthMin: 64,
		OutputLengthMax: 64,
	}

	ragCfg := sim.TenantWorkloadConfig{
		TenantID:        "rag",
		RequestCount:    500,
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
		{"round-robin", func() policy.Policy { return policy.NewRoundRobin() }},
		{"vtc", func() policy.Policy { return policy.NewVTC(policy.DefaultVTCConfig()) }},
	}

	policyRuns := make(map[string][]struct {
		chatRawShare    float64
		ragRawShare     float64
		chatWeighted    float64
		ragWeighted     float64
		chatReqShare    float64
		ragReqShare     float64
		chatTTFTMean    float64
		chatTTFTP95     float64
		ragTTFTMean     float64
		ragTTFTP95      float64
		backloggedSteps float64
		totalRawTokens  float64
	})

	for s := int64(0); s < numSeeds; s++ {
		seed := seedStart + s
		gen := sim.NewWorkloadGenerator(seed)
		trace := gen.GenerateTrace([]sim.TenantWorkloadConfig{chatCfg, ragCfg})

		for _, p := range policies {
			pol := p.fn()
			qm := sim.NewTenantQueueManager()
			server := sim.NewServer(serverCfg)

			var nowUs int64 = 0
			traceIdx := 0

			var rawChat, rawRAG int
			var weightedChat, weightedRAG float64
			var compChat, compRAG int
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

						for _, comp := range res.Completed {
							if comp.TenantID() == "chat" {
								compChat++
							} else {
								compRAG++
							}
						}
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
			totalReqs := float64(compChat + compRAG)

			var cRawShare, rRawShare, cWShare, rWShare, cReqShare, rReqShare float64
			if totalRaw > 0 {
				cRawShare = float64(rawChat) / totalRaw
				rRawShare = float64(rawRAG) / totalRaw
			}
			if totalWeighted > 0 {
				cWShare = weightedChat / totalWeighted
				rWShare = weightedRAG / totalWeighted
			}
			if totalReqs > 0 {
				cReqShare = float64(compChat) / totalReqs
				rReqShare = float64(compRAG) / totalReqs
			}

			cPerc := sim.CalculatePercentiles(chatTTFTs)
			rPerc := sim.CalculatePercentiles(ragTTFTs)

			policyRuns[p.name] = append(policyRuns[p.name], struct {
				chatRawShare    float64
				ragRawShare     float64
				chatWeighted    float64
				ragWeighted     float64
				chatReqShare    float64
				ragReqShare     float64
				chatTTFTMean    float64
				chatTTFTP95     float64
				ragTTFTMean     float64
				ragTTFTP95      float64
				backloggedSteps float64
				totalRawTokens  float64
			}{
				chatRawShare:    cRawShare,
				ragRawShare:     rRawShare,
				chatWeighted:    cWShare,
				ragWeighted:     rWShare,
				chatReqShare:    cReqShare,
				ragReqShare:     rReqShare,
				chatTTFTMean:    cPerc.Mean / 1000.0,
				chatTTFTP95:     cPerc.P95 / 1000.0,
				ragTTFTMean:     rPerc.Mean / 1000.0,
				ragTTFTP95:      rPerc.P95 / 1000.0,
				backloggedSteps: float64(bothSteps),
				totalRawTokens:  totalRaw,
			})
		}
	}

	var results []Exp2PolicyResult
	for _, p := range policies {
		runs := policyRuns[p.name]
		cRaws := make([]float64, len(runs))
		rRaws := make([]float64, len(runs))
		cWeights := make([]float64, len(runs))
		rWeights := make([]float64, len(runs))
		cReqs := make([]float64, len(runs))
		rReqs := make([]float64, len(runs))
		cTTFTMeans := make([]float64, len(runs))
		cTTFTP95s := make([]float64, len(runs))
		rTTFTMeans := make([]float64, len(runs))
		rTTFTP95s := make([]float64, len(runs))
		steps := make([]float64, len(runs))
		tokens := make([]float64, len(runs))

		for i, r := range runs {
			cRaws[i] = r.chatRawShare
			rRaws[i] = r.ragRawShare
			cWeights[i] = r.chatWeighted
			rWeights[i] = r.ragWeighted
			cReqs[i] = r.chatReqShare
			rReqs[i] = r.ragReqShare
			cTTFTMeans[i] = r.chatTTFTMean
			cTTFTP95s[i] = r.chatTTFTP95
			rTTFTMeans[i] = r.ragTTFTMean
			rTTFTP95s[i] = r.ragTTFTP95
			steps[i] = r.backloggedSteps
			tokens[i] = r.totalRawTokens
		}

		res := Exp2PolicyResult{
			PolicyName:           p.name,
			ChatRawTokenShare:    computeSummary(cRaws),
			RAGRawTokenShare:     computeSummary(rRaws),
			ChatWeightedShare:    computeSummary(cWeights),
			RAGWeightedShare:     computeSummary(rWeights),
			ChatRequestShare:     computeSummary(cReqs),
			RAGRequestShare:      computeSummary(rReqs),
			ChatTTFTMeanMs:       computeSummary(cTTFTMeans),
			ChatTTFTP95Ms:        computeSummary(cTTFTP95s),
			RAGTTFTMeanMs:        computeSummary(rTTFTMeans),
			RAGTTFTP95Ms:         computeSummary(rTTFTP95s),
			BothBackloggedSteps:  computeSummary(steps),
			TotalWindowRawTokens: computeSummary(tokens),
		}
		results = append(results, res)
	}

	outDir := filepath.Join("experiments", "results")
	_ = os.MkdirAll(outDir, 0755)

	sumFile := Exp2SummaryFile{
		Experiment:  "exp_heterogeneous_steady_state",
		Environment: getEnvMetadata(),
		Policies:    results,
	}

	jsonBytes, _ := json.MarshalIndent(sumFile, "", "  ")
	_ = os.WriteFile(filepath.Join(outDir, "exp_heterogeneous_summary.json"), jsonBytes, 0644)

	// Print Markdown Table
	fmt.Printf("\n=== EXPERIMENT 2: STEADY-STATE TOKEN SHARES (N=%d seeds, 95%% CI) ===\n", numSeeds)
	fmt.Printf("Window: [10.0s, 60.0s], conditioned on BOTH queues backlogged\n")
	fmt.Printf("| %-14s | %-26s | %-26s |\n",
		"Policy", "Chat Weighted Share (w=1,2)", "Chat Raw Token Share")
	fmt.Printf("|----------------|----------------------------|----------------------------|\n")

	for _, r := range results {
		fmt.Printf("| %-14s | %5.1f%% [%4.1f%%, %4.1f%%]    | %5.1f%% [%4.1f%%, %4.1f%%]    |\n",
			r.PolicyName,
			r.ChatWeightedShare.Mean*100, r.ChatWeightedShare.CI95Low*100, r.ChatWeightedShare.CI95Hi*100,
			r.ChatRawTokenShare.Mean*100, r.ChatRawTokenShare.CI95Low*100, r.ChatRawTokenShare.CI95Hi*100,
		)
	}

	fmt.Printf("\nComparison with Hand-Calculated Predictions:\n")
	fmt.Printf("| %-14s | %-16s | %-16s | %-16s | %-16s |\n",
		"Policy", "Hand-Calc Wtd", "Measured Wtd CI", "Hand-Calc Raw", "Measured Raw CI")
	fmt.Printf("|----------------|------------------|------------------|------------------|------------------|\n")
	fmt.Printf("| %-14s | ~0.0%% - 5.0%%   | %5.1f%% [%4.1f-%4.1f] | <5.0%%           | %5.1f%% [%4.1f-%4.1f] |\n",
		"fifo",
		results[0].ChatWeightedShare.Mean*100, results[0].ChatWeightedShare.CI95Low*100, results[0].ChatWeightedShare.CI95Hi*100,
		results[0].ChatRawTokenShare.Mean*100, results[0].ChatRawTokenShare.CI95Low*100, results[0].ChatRawTokenShare.CI95Hi*100,
	)
	fmt.Printf("| %-14s | ~5.7%% (1:1 req) | %5.1f%% [%4.1f-%4.1f] | ~4.4%% (1:1 req) | %5.1f%% [%4.1f-%4.1f] |\n",
		"round-robin",
		results[1].ChatWeightedShare.Mean*100, results[1].ChatWeightedShare.CI95Low*100, results[1].ChatWeightedShare.CI95Hi*100,
		results[1].ChatRawTokenShare.Mean*100, results[1].ChatRawTokenShare.CI95Low*100, results[1].ChatRawTokenShare.CI95Hi*100,
	)
	fmt.Printf("| %-14s | ~50.0%%          | %5.1f%% [%4.1f-%4.1f] | ~43.6%%          | %5.1f%% [%4.1f-%4.1f] |\n",
		"vtc",
		results[2].ChatWeightedShare.Mean*100, results[2].ChatWeightedShare.CI95Low*100, results[2].ChatWeightedShare.CI95Hi*100,
		results[2].ChatRawTokenShare.Mean*100, results[2].ChatRawTokenShare.CI95Low*100, results[2].ChatRawTokenShare.CI95Hi*100,
	)

	fmt.Printf("\nTTFT Performance (Mean / P95):\n")
	fmt.Printf("| %-14s | %-22s | %-22s |\n",
		"Policy", "Chat TTFT (Mean / P95)", "RAG TTFT (Mean / P95)")
	fmt.Printf("|----------------|------------------------|------------------------|\n")
	for _, r := range results {
		fmt.Printf("| %-14s | %6.1f ms / %6.1f ms   | %6.1f ms / %6.1f ms   |\n",
			r.PolicyName,
			r.ChatTTFTMeanMs.Mean, r.ChatTTFTP95Ms.Mean,
			r.RAGTTFTMeanMs.Mean, r.RAGTTFTP95Ms.Mean,
		)
	}
}
