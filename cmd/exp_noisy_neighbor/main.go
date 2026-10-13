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

type SeedResult struct {
	Seed               int64   `json:"seed"`
	Policy             string  `json:"policy"`
	VictimSoloLatMean  float64 `json:"victim_solo_lat_mean"`
	VictimColocLatMean float64 `json:"victim_coloc_lat_mean"`
	VictimSlowdown     float64 `json:"victim_slowdown"`
	VictimTTFTMean     float64 `json:"victim_ttft_mean"`
	VictimTTFTP95      float64 `json:"victim_ttft_p95"`
	VictimITLMean      float64 `json:"victim_itl_mean"`
	FlooderLatMean     float64 `json:"flooder_lat_mean"`
	JainsIndexSliding  float64 `json:"jains_index_sliding"`
	GoodputTokensSec   float64 `json:"goodput_tokens_sec"`
	TotalTokensServed  int     `json:"total_tokens_served"`
	IdleWhileQueuedUs  int64   `json:"idle_while_queued_us"`
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

type PolicySummary struct {
	PolicyName         string       `json:"policy_name"`
	VictimSlowdown     SummaryStats `json:"victim_slowdown"`
	VictimLatencyUs    SummaryStats `json:"victim_latency_us"`
	VictimTTFTUs       SummaryStats `json:"victim_ttft_us"`
	VictimITLUs        SummaryStats `json:"victim_itl_us"`
	JainsIndexSliding  SummaryStats `json:"jains_index_sliding"`
	GoodputTokensSec   SummaryStats `json:"goodput_tokens_sec"`
	IdleWhileQueuedUs  int64        `json:"idle_while_queued_us"`
}

func main() {
	const (
		numSeeds            = 30
		seedStart     int64 = 100
		slidingWindow       = 1000
	)

	serverCfg := sim.ServerConfig{
		KVCapacityTokens: 4096,
		MaxBatchSize:     16,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	victimCfg := sim.TenantWorkloadConfig{
		TenantID:        "victim",
		RequestCount:    60,
		ArrivalProcess:  "poisson",
		MeanIntervalUs:  50000, // 50 ms
		PromptLengthMin: 128,
		PromptLengthMax: 128,
		OutputLengthMin: 64,
		OutputLengthMax: 64,
	}

	flooderCfg := sim.TenantWorkloadConfig{
		TenantID:        "flooder",
		RequestCount:    240,
		ArrivalProcess:  "poisson",
		MeanIntervalUs:  12000, // 12 ms
		PromptLengthMin: 1024,
		PromptLengthMax: 1024,
		OutputLengthMin: 256,
		OutputLengthMax: 256,
	}

	policyFactories := []struct {
		name string
		fn   func() policy.Policy
	}{
		{
			name: "fifo",
			fn:   func() policy.Policy { return policy.NewFIFO() },
		},
		{
			name: "round-robin",
			fn:   func() policy.Policy { return policy.NewRoundRobin() },
		},
		{
			name: "vtc-default",
			fn: func() policy.Policy {
				cfg := policy.DefaultVTCConfig() // artifact lift, incremental charging, w_in=1, w_out=2
				return policy.NewVTC(cfg)
			},
		},
		{
			name: "vtc-paper-lift",
			fn: func() policy.Policy {
				cfg := policy.DefaultVTCConfig()
				cfg.LiftMode = policy.LiftModePaper
				return policy.NewVTC(cfg)
			},
		},
		{
			name: "vtc-completion-charge",
			fn: func() policy.Policy {
				cfg := policy.DefaultVTCConfig()
				cfg.ChargingModel = policy.ChargingCompletion
				return policy.NewVTC(cfg)
			},
		},
		{
			name: "vtc-wout-1",
			fn: func() policy.Policy {
				cfg := policy.DefaultVTCConfig()
				cfg.OutputWeight = 1.0
				return policy.NewVTC(cfg)
			},
		},
		{
			name: "vtc-wout-2",
			fn: func() policy.Policy {
				cfg := policy.DefaultVTCConfig()
				cfg.OutputWeight = 2.0
				return policy.NewVTC(cfg)
			},
		},
		{
			name: "vtc-wout-3",
			fn: func() policy.Policy {
				cfg := policy.DefaultVTCConfig()
				cfg.OutputWeight = 3.0
				return policy.NewVTC(cfg)
			},
		},
		{
			name: "vtc-wout-4",
			fn: func() policy.Policy {
				cfg := policy.DefaultVTCConfig()
				cfg.OutputWeight = 4.0
				return policy.NewVTC(cfg)
			},
		},
	}

	var allSeedResults []SeedResult
	policyResultsMap := make(map[string][]SeedResult)

	for seedOffset := int64(0); seedOffset < numSeeds; seedOffset++ {
		seed := seedStart + seedOffset
		gen := sim.NewWorkloadGenerator(seed)

		// 1. Generate solo trace for Victim alone to measure true solo baseline
		soloTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{victimCfg})
		soloSim := sim.NewSimulator(serverCfg, policy.NewFIFO(), slidingWindow)
		soloMetrics := soloSim.Run(soloTrace)
		victimSoloLat := soloMetrics.PerTenant["victim"].TotalLatencyUs.Mean

		// 2. Generate co-located trace for Victim + Flooder
		colocTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{victimCfg, flooderCfg})

		// Run each policy on the identical co-located trace
		for _, pf := range policyFactories {
			pol := pf.fn()
			colocSim := sim.NewSimulator(serverCfg, pol, slidingWindow)
			colocMetrics := colocSim.Run(colocTrace)

			vMet := colocMetrics.PerTenant["victim"]
			fMet := colocMetrics.PerTenant["flooder"]

			victimColocLat := vMet.TotalLatencyUs.Mean
			slowdown := victimColocLat / victimSoloLat

			sr := SeedResult{
				Seed:               seed,
				Policy:             pf.name,
				VictimSoloLatMean:  victimSoloLat,
				VictimColocLatMean: victimColocLat,
				VictimSlowdown:     slowdown,
				VictimTTFTMean:     vMet.TTFTUs.Mean,
				VictimTTFTP95:      vMet.TTFTUs.P95,
				VictimITLMean:      vMet.ITLUs.Mean,
				FlooderLatMean:     fMet.TotalLatencyUs.Mean,
				JainsIndexSliding:  colocMetrics.JainsIndexSliding,
				GoodputTokensSec:   colocMetrics.GoodputTokensPerSec,
				TotalTokensServed:  colocMetrics.TotalTokensServed,
				IdleWhileQueuedUs:  colocMetrics.IdleWhileQueuedUs,
			}

			allSeedResults = append(allSeedResults, sr)
			policyResultsMap[pf.name] = append(policyResultsMap[pf.name], sr)
		}
	}

	// Compute summaries per policy
	var summaries []PolicySummary
	for _, pf := range policyFactories {
		runs := policyResultsMap[pf.name]
		slowdowns := make([]float64, len(runs))
		lats := make([]float64, len(runs))
		ttfts := make([]float64, len(runs))
		itls := make([]float64, len(runs))
		jains := make([]float64, len(runs))
		goodputs := make([]float64, len(runs))
		var maxIdle int64 = 0

		for i, r := range runs {
			slowdowns[i] = r.VictimSlowdown
			lats[i] = r.VictimColocLatMean
			ttfts[i] = r.VictimTTFTMean
			itls[i] = r.VictimITLMean
			jains[i] = r.JainsIndexSliding
			goodputs[i] = r.GoodputTokensSec
			if r.IdleWhileQueuedUs > maxIdle {
				maxIdle = r.IdleWhileQueuedUs
			}
		}

		ps := PolicySummary{
			PolicyName:         pf.name,
			VictimSlowdown:     computeSummary(slowdowns),
			VictimLatencyUs:    computeSummary(lats),
			VictimTTFTUs:       computeSummary(ttfts),
			VictimITLUs:        computeSummary(itls),
			JainsIndexSliding:  computeSummary(jains),
			GoodputTokensSec:   computeSummary(goodputs),
			IdleWhileQueuedUs:  maxIdle,
		}
		summaries = append(summaries, ps)
	}

	// Output directory
	outDir := filepath.Join("experiments", "results")
	_ = os.MkdirAll(outDir, 0755)

	// Save raw seed results JSON
	rawPath := filepath.Join(outDir, "exp_noisy_neighbor_raw.json")
	rawBytes, _ := json.MarshalIndent(allSeedResults, "", "  ")
	_ = os.WriteFile(rawPath, rawBytes, 0644)

	type Exp1SummaryFile struct {
		Experiment  string              `json:"experiment"`
		Environment EnvironmentMetadata `json:"environment"`
		Policies    []PolicySummary     `json:"policies"`
	}

	sumFile := Exp1SummaryFile{
		Experiment:  "exp_noisy_neighbor",
		Environment: getEnvMetadata(),
		Policies:    summaries,
	}

	// Save summary JSON
	summaryPath := filepath.Join(outDir, "exp_noisy_neighbor_summary.json")
	sumBytes, _ := json.MarshalIndent(sumFile, "", "  ")
	_ = os.WriteFile(summaryPath, sumBytes, 0644)

	// Print high-visibility Markdown table
	fmt.Printf("\n=== NOISY NEIGHBOR EXPERIMENT RESULTS (N=%d seeds, 95%% Bootstrap CI) ===\n", numSeeds)
	fmt.Printf("| %-22s | %-16s | %-20s | %-16s | %-14s |\n",
		"Policy", "Victim Slowdown", "Victim TTFT (Mean / P95)", "Jain's Index", "Idle-Queued (µs)")
	fmt.Printf("|------------------------|------------------|----------------------|------------------|----------------|\n")

	for _, s := range summaries {
		fmt.Printf("| %-22s | %5.2fx [%4.2f, %4.2f] | %5.1f ms / %5.1f ms  | %5.3f [%4.3f, %4.3f] | %14d |\n",
			s.PolicyName,
			s.VictimSlowdown.Mean, s.VictimSlowdown.CI95Low, s.VictimSlowdown.CI95Hi,
			s.VictimTTFTUs.Mean/1000.0, s.VictimTTFTUs.P95/1000.0,
			s.JainsIndexSliding.Mean, s.JainsIndexSliding.CI95Low, s.JainsIndexSliding.CI95Hi,
			s.IdleWhileQueuedUs,
		)
	}
	fmt.Printf("\nResults written to %s and %s\n", rawPath, summaryPath)
}
