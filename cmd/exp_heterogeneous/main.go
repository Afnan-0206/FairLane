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

	return EnvironmentMetadata{
		GitCommit:     commit,
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
	PolicyName         string       `json:"policy_name"`
	ChatTokenShare     SummaryStats `json:"chat_token_share"`
	RAGTokenShare      SummaryStats `json:"rag_token_share"`
	ChatRequestShare   SummaryStats `json:"chat_request_share"`
	RAGRequestShare    SummaryStats `json:"rag_request_share"`
	ChatTTFTMeanMs     SummaryStats `json:"chat_ttft_mean_ms"`
	ChatTTFTP95Ms      SummaryStats `json:"chat_ttft_p95_ms"`
	RAGTTFTMeanMs      SummaryStats `json:"rag_ttft_mean_ms"`
	RAGTTFTP95Ms       SummaryStats `json:"rag_ttft_p95_ms"`
	JainsIndexSliding  SummaryStats `json:"jains_index_sliding"`
	TotalTokensServed  int          `json:"total_tokens_served"`
	SimulatedDurationS float64      `json:"simulated_duration_s"`
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
	)

	serverCfg := sim.ServerConfig{
		KVCapacityTokens: 16384, // 16k KV to accommodate 4k RAG prompts alongside chat
		MaxBatchSize:     16,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	chatCfg := sim.TenantWorkloadConfig{
		TenantID:        "chat",
		RequestCount:    200,
		ArrivalProcess:  "periodic",
		MeanIntervalUs:  1000, // all 200 requests arrive within 200ms -> permanent backlog
		PromptLengthMin: 128,
		PromptLengthMax: 128,
		OutputLengthMin: 64,
		OutputLengthMax: 64,
	}

	ragCfg := sim.TenantWorkloadConfig{
		TenantID:        "rag",
		RequestCount:    50,
		ArrivalProcess:  "periodic",
		MeanIntervalUs:  4000, // all 50 requests arrive within 200ms -> permanent backlog
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
		chatTokShare float64
		ragTokShare  float64
		chatReqShare float64
		ragReqShare  float64
		chatTTFTMean float64
		chatTTFTP95  float64
		ragTTFTMean  float64
		ragTTFTP95   float64
		jains        float64
		totalTokens  int
		durationS    float64
	})

	for s := int64(0); s < numSeeds; s++ {
		seed := seedStart + s
		gen := sim.NewWorkloadGenerator(seed)
		trace := gen.GenerateTrace([]sim.TenantWorkloadConfig{chatCfg, ragCfg})

		for _, p := range policies {
			pol := p.fn()
			simulator := sim.NewSimulator(serverCfg, pol, 2000)
			met := simulator.Run(trace)

			chatMet := met.PerTenant["chat"]
			ragMet := met.PerTenant["rag"]

			totalTok := float64(chatMet.TotalTokensServed + ragMet.TotalTokensServed)
			totalReq := float64(chatMet.RequestsCompleted + ragMet.RequestsCompleted)

			chatTokShare := float64(chatMet.TotalTokensServed) / totalTok
			ragTokShare := float64(ragMet.TotalTokensServed) / totalTok
			chatReqShare := float64(chatMet.RequestsCompleted) / totalReq
			ragReqShare := float64(ragMet.RequestsCompleted) / totalReq

			policyRuns[p.name] = append(policyRuns[p.name], struct {
				chatTokShare float64
				ragTokShare  float64
				chatReqShare float64
				ragReqShare  float64
				chatTTFTMean float64
				chatTTFTP95  float64
				ragTTFTMean  float64
				ragTTFTP95   float64
				jains        float64
				totalTokens  int
				durationS    float64
			}{
				chatTokShare: chatTokShare,
				ragTokShare:  ragTokShare,
				chatReqShare: chatReqShare,
				ragReqShare:  ragReqShare,
				chatTTFTMean: chatMet.TTFTUs.Mean / 1000.0,
				chatTTFTP95:  chatMet.TTFTUs.P95 / 1000.0,
				ragTTFTMean:  ragMet.TTFTUs.Mean / 1000.0,
				ragTTFTP95:   ragMet.TTFTUs.P95 / 1000.0,
				jains:        met.JainsIndexSliding,
				totalTokens:  met.TotalTokensServed,
				durationS:    float64(met.TotalVirtualTimeUs) / 1e6,
			})
		}
	}

	var results []Exp2PolicyResult
	for _, p := range policies {
		runs := policyRuns[p.name]
		cToks := make([]float64, len(runs))
		rToks := make([]float64, len(runs))
		cReqs := make([]float64, len(runs))
		rReqs := make([]float64, len(runs))
		cTTFTMeans := make([]float64, len(runs))
		cTTFTP95s := make([]float64, len(runs))
		rTTFTMeans := make([]float64, len(runs))
		rTTFTP95s := make([]float64, len(runs))
		jains := make([]float64, len(runs))

		for i, r := range runs {
			cToks[i] = r.chatTokShare
			rToks[i] = r.ragTokShare
			cReqs[i] = r.chatReqShare
			rReqs[i] = r.ragReqShare
			cTTFTMeans[i] = r.chatTTFTMean
			cTTFTP95s[i] = r.chatTTFTP95
			rTTFTMeans[i] = r.ragTTFTMean
			rTTFTP95s[i] = r.ragTTFTP95
			jains[i] = r.jains
		}

		res := Exp2PolicyResult{
			PolicyName:         p.name,
			ChatTokenShare:     computeSummary(cToks),
			RAGTokenShare:      computeSummary(rToks),
			ChatRequestShare:   computeSummary(cReqs),
			RAGRequestShare:    computeSummary(rReqs),
			ChatTTFTMeanMs:     computeSummary(cTTFTMeans),
			ChatTTFTP95Ms:      computeSummary(cTTFTP95s),
			RAGTTFTMeanMs:      computeSummary(rTTFTMeans),
			RAGTTFTP95Ms:       computeSummary(rTTFTP95s),
			JainsIndexSliding:  computeSummary(jains),
			TotalTokensServed:  runs[0].totalTokens,
			SimulatedDurationS: runs[0].durationS,
		}
		results = append(results, res)
	}

	outDir := filepath.Join("experiments", "results")
	_ = os.MkdirAll(outDir, 0755)

	sumFile := Exp2SummaryFile{
		Experiment:  "exp_heterogeneous",
		Environment: getEnvMetadata(),
		Policies:    results,
	}

	jsonBytes, _ := json.MarshalIndent(sumFile, "", "  ")
	_ = os.WriteFile(filepath.Join(outDir, "exp_heterogeneous_summary.json"), jsonBytes, 0644)

	// Print Markdown Table
	fmt.Printf("\n=== EXPERIMENT 2: HETEROGENEOUS REQUEST SIZING (N=30 seeds, 95%% CI) ===\n")
	fmt.Printf("| %-14s | %-24s | %-24s | %-16s |\n",
		"Policy", "Chat Token Share (%)", "RAG Token Share (%)", "Jain's Index")
	fmt.Printf("|----------------|--------------------------|--------------------------|------------------|\n")

	for _, r := range results {
		fmt.Printf("| %-14s | %5.1f%% [%4.1f%%, %4.1f%%]  | %5.1f%% [%4.1f%%, %4.1f%%]  | %5.3f [%4.3f, %4.3f] |\n",
			r.PolicyName,
			r.ChatTokenShare.Mean*100, r.ChatTokenShare.CI95Low*100, r.ChatTokenShare.CI95Hi*100,
			r.RAGTokenShare.Mean*100, r.RAGTokenShare.CI95Low*100, r.RAGTokenShare.CI95Hi*100,
			r.JainsIndexSliding.Mean, r.JainsIndexSliding.CI95Low, r.JainsIndexSliding.CI95Hi,
		)
	}

	fmt.Printf("\nPer-Tenant Request Shares and Latencies:\n")
	fmt.Printf("| %-14s | %-16s | %-16s | %-18s | %-18s |\n",
		"Policy", "Chat Req Share", "RAG Req Share", "Chat TTFT Mean (ms)", "Chat TTFT P95 (ms)")
	fmt.Printf("|----------------|------------------|------------------|--------------------|--------------------|\n")
	for _, r := range results {
		fmt.Printf("| %-14s | %5.1f%%           | %5.1f%%           | %7.1f ms          | %7.1f ms          |\n",
			r.PolicyName,
			r.ChatRequestShare.Mean*100, r.RAGRequestShare.Mean*100,
			r.ChatTTFTMeanMs.Mean, r.ChatTTFTP95Ms.Mean,
		)
	}
}
