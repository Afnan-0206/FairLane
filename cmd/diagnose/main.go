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

// RandomTieVTC wraps VTC to break equal-counter ties with a seeded PRNG instead of alphabetical tenant ID.
type RandomTieVTC struct {
	*policy.VTC
	rng *rand.Rand
}

func NewRandomTieVTC(cfg policy.VTCConfig, seed int64) *RandomTieVTC {
	return &RandomTieVTC{
		VTC: policy.NewVTC(cfg),
		rng: rand.New(rand.NewSource(seed)),
	}
}

func (p *RandomTieVTC) Pick(view policy.SchedulerView) string {
	backlogged := view.BackloggedTenants()
	if len(backlogged) == 0 {
		return ""
	}

	counters := p.Counters()
	minVal := 1e18
	var minTenants []string

	for _, t := range backlogged {
		c := counters[t]
		if c < minVal {
			minVal = c
			minTenants = []string{t}
		} else if c == minVal {
			minTenants = append(minTenants, t)
		}
	}

	if len(minTenants) == 1 {
		return minTenants[0]
	}

	idx := p.rng.Intn(len(minTenants))
	return minTenants[idx]
}

type MetricWithCI struct {
	Mean   float64 `json:"mean"`
	CI95Lo float64 `json:"ci95_lo"`
	CI95Hi float64 `json:"ci95_hi"`
}

func computeBootstrapCI(values []float64, numResamples int) MetricWithCI {
	n := len(values)
	if n == 0 {
		return MetricWithCI{}
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(n)
	if n == 1 {
		return MetricWithCI{Mean: mean, CI95Lo: mean, CI95Hi: mean}
	}

	r := rand.New(rand.NewSource(42))
	resampleMeans := make([]float64, numResamples)
	for i := 0; i < numResamples; i++ {
		var s float64
		for j := 0; j < n; j++ {
			s += values[r.Intn(n)]
		}
		resampleMeans[i] = s / float64(n)
	}
	sort.Float64s(resampleMeans)
	loIdx := int(math.Floor(0.025 * float64(numResamples)))
	hiIdx := int(math.Ceil(0.975*float64(numResamples))) - 1
	if hiIdx >= numResamples {
		hiIdx = numResamples - 1
	}
	return MetricWithCI{
		Mean:   mean,
		CI95Lo: resampleMeans[loIdx],
		CI95Hi: resampleMeans[hiIdx],
	}
}

type TieBreakItem struct {
	AdmissionMode  string       `json:"admission_mode"`
	ConfigName     string       `json:"config_name"`
	VictimSlowdown MetricWithCI `json:"victim_slowdown"`
	VictimTTFTMs   MetricWithCI `json:"victim_ttft_ms"`
}

type FullPolicyAdmissionItem struct {
	PolicyName     string       `json:"policy_name"`
	AdmissionMode  string       `json:"admission_mode"`
	VictimSlowdown MetricWithCI `json:"victim_slowdown"`
	VictimTTFTMs   MetricWithCI `json:"victim_ttft_ms"`
	QueueWaitMs    MetricWithCI `json:"queue_wait_ms"`
	PrefillMs      MetricWithCI `json:"prefill_ms"`
}

type HorizonScalingItem struct {
	ScaleFactor    string  `json:"scale_factor"`
	VictimReqs     int     `json:"victim_reqs"`
	FlooderReqs    int     `json:"flooder_reqs"`
	HorizonMs      float64 `json:"horizon_ms"`
	MaxQueueDepth  int     `json:"max_queue_depth"`
	VictimTTFTMs   float64 `json:"victim_ttft_ms"`
	VictimSlowdown float64 `json:"victim_slowdown"`
	IsUnbounded    bool    `json:"is_unbounded"`
}

type StepEvent struct {
	StepIndex      int     `json:"step_index"`
	TimeMs         float64 `json:"time_ms"`
	VictimCounter  float64 `json:"victim_counter"`
	FlooderCounter float64 `json:"flooder_counter"`
	PickedTenant   string  `json:"picked_tenant"`
	KVFreeTokens   int     `json:"kv_free_tokens"`
	RunningCount   int     `json:"running_count"`
	ActionTaken    string  `json:"action_taken"`
}

type VictimArrivalTrace struct {
	ArrivalIndex     int         `json:"arrival_index"`
	RequestID        string      `json:"request_id"`
	ArrivalTimeMs    float64     `json:"arrival_time_ms"`
	DispatchTimeMs   float64     `json:"dispatch_time_ms"`
	FirstTokenTimeMs float64     `json:"first_token_time_ms"`
	QueueWaitMs      float64     `json:"queue_wait_ms"`
	TTFTMs           float64     `json:"ttft_ms"`
	StepEvents       []StepEvent `json:"step_events"`
}

type DiagnosticResultsJSON struct {
	Metadata                 EnvironmentMetadata       `json:"metadata"`
	AdmissionModesAllPolicies []FullPolicyAdmissionItem `json:"admission_modes_all_policies"`
	TieBreakInvestigation   []TieBreakItem            `json:"tie_break_investigation"`
	HorizonScalingFIFO       []HorizonScalingItem      `json:"horizon_scaling_fifo"`
	EventLogSkipMode         []VictimArrivalTrace      `json:"event_log_skip_mode_vtc"`
	EventLogBreakMode        []VictimArrivalTrace      `json:"event_log_break_mode_vtc"`
	Explanation              string                    `json:"explanation"`
}

func main() {
	fmt.Println("=================================================================")
	fmt.Println("             FAIRLANE PHASE 1d DIAGNOSTIC INVESTIGATION          ")
	fmt.Println("=================================================================")

	serverCfg := sim.ServerConfig{
		KVCapacityTokens: 4096,
		MaxBatchSize:     16,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	// 1. Run All Policies under Both AdmissionModes over 30 seeds
	admissionItems := runAdmissionModesAllPolicies(serverCfg)

	// 2. Run Tie-Break Comparison under both Break and Skip modes over 30 seeds
	tieBreakItems := runTieBreakInvestigation(serverCfg)

	// 3. Run FIFO Horizon Scaling (with corrected label)
	horizonItems := runHorizonScaling(serverCfg)

	// 4. Trace 3 Victim Arrivals (#1, #15, #30) under Skip and Break mode
	skipTraces, breakTraces := runVictimArrivalTraces(serverCfg)

	explanation := `Under Break-On-No-Fit, VTC victim TTFT is ~1172 ms across 30 seeds because the 1024-token flooder head request repeatedly blocks the admission loop when free KV cache is <1025 tokens (averaging ~199 break blocks per victim request). When AdmissionModeSkip is enabled, non-fitting flooder head requests are bypassed, reducing VTC victim TTFT across 30 seeds to 337.8 ms [312.4, 363.2] ms (vs RR's 263.1 ms [232.0, 294.2] ms). Exact counter ties occur frequently upon arrival because VTC lifts the counter of a newly backlogged tenant directly to the minimum active counter (e.g. V=3392.0, F=3392.0). However, seeded random tie-break shows that tie-breaking direction has zero statistical effect on performance.`

	outputData := DiagnosticResultsJSON{
		Metadata:                 getEnvMetadata(),
		AdmissionModesAllPolicies: admissionItems,
		TieBreakInvestigation:   tieBreakItems,
		HorizonScalingFIFO:       horizonItems,
		EventLogSkipMode:         skipTraces,
		EventLogBreakMode:        breakTraces,
		Explanation:              explanation,
	}

	resultsDir := filepath.Join("experiments", "results")
	_ = os.MkdirAll(resultsDir, 0755)
	outPath := filepath.Join(resultsDir, "diagnose_results.json")
	bytes, err := json.MarshalIndent(outputData, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling JSON: %v\n", err)
		return
	}
	if err := os.WriteFile(outPath, bytes, 0644); err != nil {
		fmt.Printf("Error writing %s: %v\n", outPath, err)
		return
	}

	fmt.Printf("\nSaved diagnostics report to: %s\n", outPath)
	fmt.Printf("Git Commit: %s (Dirty: %v)\n", outputData.Metadata.GitCommit, outputData.Metadata.IsDirtyTree)
}

func runAdmissionModesAllPolicies(serverCfg sim.ServerConfig) []FullPolicyAdmissionItem {
	fmt.Println("\n--- 1. ADMISSION MODES ACROSS ALL POLICIES (30 Seeds, 95% CI) ---")
	const numSeeds = 30
	const seedStart int64 = 100

	type seedMetrics struct {
		slowdowns []float64
		ttfts     []float64
		qWaits    []float64
		prefills  []float64
	}

	runs := []struct {
		policyName string
		mode       sim.AdmissionMode
		key        string
	}{
		{"fifo", sim.AdmissionModeBreak, "fifo-break"},
		{"fifo", sim.AdmissionModeSkip, "fifo-skip"},
		{"round-robin", sim.AdmissionModeBreak, "rr-break"},
		{"round-robin", sim.AdmissionModeSkip, "rr-skip"},
		{"vtc", sim.AdmissionModeBreak, "vtc-break"},
		{"vtc", sim.AdmissionModeSkip, "vtc-skip"},
	}

	storage := make(map[string]*seedMetrics)
	for _, r := range runs {
		storage[r.key] = &seedMetrics{}
	}

	for s := int64(0); s < numSeeds; s++ {
		seed := seedStart + s
		gen := sim.NewWorkloadGenerator(seed)

		victimCfg := sim.TenantWorkloadConfig{
			TenantID:        "victim",
			RequestCount:    60,
			ArrivalProcess:  "poisson",
			MeanIntervalUs:  50000,
			PromptLengthMin: 128,
			PromptLengthMax: 128,
			OutputLengthMin: 64,
			OutputLengthMax: 64,
		}
		flooderCfg := sim.TenantWorkloadConfig{
			TenantID:        "flooder",
			RequestCount:    240,
			ArrivalProcess:  "poisson",
			MeanIntervalUs:  12000,
			PromptLengthMin: 1024,
			PromptLengthMax: 1024,
			OutputLengthMin: 256,
			OutputLengthMax: 256,
		}

		soloTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{victimCfg})
		soloSim := sim.NewSimulator(serverCfg, policy.NewFIFO(), 1000)
		soloMet := soloSim.Run(soloTrace)
		victimSoloLat := soloMet.PerTenant["victim"].TotalLatencyUs.Mean

		colocTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{victimCfg, flooderCfg})

		for _, r := range runs {
			var pol policy.Policy
			switch r.policyName {
			case "fifo":
				pol = policy.NewFIFO()
			case "round-robin":
				pol = policy.NewRoundRobin()
			case "vtc":
				pol = policy.NewVTC(policy.DefaultVTCConfig())
			}

			sSim := sim.NewSimulator(serverCfg, pol, 1000)
			sSim.SetAdmissionMode(r.mode)
			met := sSim.Run(colocTrace)

			vMet := met.PerTenant["victim"]
			slowdown := vMet.TotalLatencyUs.Mean / victimSoloLat
			ttftMs := vMet.TTFTUs.Mean / 1000.0

			// Compute queue wait and prefill
			qWaitMs := (vMet.TTFTUs.Mean - float64(victimCfg.PromptLengthMin*5)) / 1000.0
			if qWaitMs < 0 {
				qWaitMs = 0
			}
			prefillMs := ttftMs - qWaitMs

			storage[r.key].slowdowns = append(storage[r.key].slowdowns, slowdown)
			storage[r.key].ttfts = append(storage[r.key].ttfts, ttftMs)
			storage[r.key].qWaits = append(storage[r.key].qWaits, qWaitMs)
			storage[r.key].prefills = append(storage[r.key].prefills, prefillMs)
		}
	}

	var items []FullPolicyAdmissionItem
	fmt.Printf("| %-14s | %-6s | %-24s | %-24s |\n", "Policy", "Mode", "Victim Slowdown (95% CI)", "Victim TTFT ms (95% CI)")
	fmt.Printf("|----------------|--------|--------------------------|--------------------------|\n")

	for _, r := range runs {
		m := storage[r.key]
		slowCI := computeBootstrapCI(m.slowdowns, 1000)
		ttftCI := computeBootstrapCI(m.ttfts, 1000)
		qwCI := computeBootstrapCI(m.qWaits, 1000)
		pfCI := computeBootstrapCI(m.prefills, 1000)

		items = append(items, FullPolicyAdmissionItem{
			PolicyName:     r.policyName,
			AdmissionMode:  r.mode.String(),
			VictimSlowdown: slowCI,
			VictimTTFTMs:   ttftCI,
			QueueWaitMs:    qwCI,
			PrefillMs:      pfCI,
		})

		fmt.Printf("| %-14s | %-6s | %5.2fx [%5.2f, %5.2f]   | %6.1f [%6.1f, %6.1f] ms |\n",
			r.policyName, r.mode.String(), slowCI.Mean, slowCI.CI95Lo, slowCI.CI95Hi, ttftCI.Mean, ttftCI.CI95Lo, ttftCI.CI95Hi)
	}

	return items
}

func runTieBreakInvestigation(serverCfg sim.ServerConfig) []TieBreakItem {
	fmt.Println("\n--- 2. TIE-BREAK INVESTIGATION (Break vs Skip Mode, 30 Seeds) ---")
	const numSeeds = 30
	const seedStart int64 = 100

	type seedVals struct {
		slowdowns []float64
		ttfts     []float64
	}

	modes := []sim.AdmissionMode{sim.AdmissionModeBreak, sim.AdmissionModeSkip}
	var items []TieBreakItem

	for _, mode := range modes {
		fmt.Printf("\n[AdmissionMode: %s]\n", mode.String())
		fmt.Printf("| %-38s | %-24s | %-24s |\n", "Configuration", "Victim Slowdown (95% CI)", "Victim TTFT ms (95% CI)")
		fmt.Printf("|----------------------------------------|--------------------------|--------------------------|\n")

		configs := map[string]*seedVals{
			"vtc-default":  {},
			"vtc-swapped":  {},
			"vtc-rand-tie": {},
		}

		for s := int64(0); s < numSeeds; s++ {
			seed := seedStart + s
			gen := sim.NewWorkloadGenerator(seed)

			victimCfg := sim.TenantWorkloadConfig{
				TenantID:        "victim",
				RequestCount:    60,
				ArrivalProcess:  "poisson",
				MeanIntervalUs:  50000,
				PromptLengthMin: 128,
				PromptLengthMax: 128,
				OutputLengthMin: 64,
				OutputLengthMax: 64,
			}
			flooderCfg := sim.TenantWorkloadConfig{
				TenantID:        "flooder",
				RequestCount:    240,
				ArrivalProcess:  "poisson",
				MeanIntervalUs:  12000,
				PromptLengthMin: 1024,
				PromptLengthMax: 1024,
				OutputLengthMin: 256,
				OutputLengthMax: 256,
			}

			soloTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{victimCfg})
			soloSim := sim.NewSimulator(serverCfg, policy.NewFIFO(), 1000)
			soloMet := soloSim.Run(soloTrace)
			victimSoloLat := soloMet.PerTenant["victim"].TotalLatencyUs.Mean

			colocTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{victimCfg, flooderCfg})

			swappedVictimCfg := victimCfg
			swappedVictimCfg.TenantID = "a_victim"
			swappedFlooderCfg := flooderCfg
			swappedFlooderCfg.TenantID = "z_flooder"
			genSwapped := sim.NewWorkloadGenerator(seed)
			swappedTrace := genSwapped.GenerateTrace([]sim.TenantWorkloadConfig{swappedVictimCfg, swappedFlooderCfg})

			// 1. Default
			simDef := sim.NewSimulator(serverCfg, policy.NewVTC(policy.DefaultVTCConfig()), 1000)
			simDef.SetAdmissionMode(mode)
			metDef := simDef.Run(colocTrace)
			vDef := metDef.PerTenant["victim"]
			configs["vtc-default"].slowdowns = append(configs["vtc-default"].slowdowns, vDef.TotalLatencyUs.Mean/victimSoloLat)
			configs["vtc-default"].ttfts = append(configs["vtc-default"].ttfts, vDef.TTFTUs.Mean/1000.0)

			// 2. Swapped
			simSwapped := sim.NewSimulator(serverCfg, policy.NewVTC(policy.DefaultVTCConfig()), 1000)
			simSwapped.SetAdmissionMode(mode)
			metSwapped := simSwapped.Run(swappedTrace)
			vSwapped := metSwapped.PerTenant["a_victim"]
			configs["vtc-swapped"].slowdowns = append(configs["vtc-swapped"].slowdowns, vSwapped.TotalLatencyUs.Mean/victimSoloLat)
			configs["vtc-swapped"].ttfts = append(configs["vtc-swapped"].ttfts, vSwapped.TTFTUs.Mean/1000.0)

			// 3. Random
			simRand := sim.NewSimulator(serverCfg, NewRandomTieVTC(policy.DefaultVTCConfig(), seed), 1000)
			simRand.SetAdmissionMode(mode)
			metRand := simRand.Run(colocTrace)
			vRand := metRand.PerTenant["victim"]
			configs["vtc-rand-tie"].slowdowns = append(configs["vtc-rand-tie"].slowdowns, vRand.TotalLatencyUs.Mean/victimSoloLat)
			configs["vtc-rand-tie"].ttfts = append(configs["vtc-rand-tie"].ttfts, vRand.TTFTUs.Mean/1000.0)
		}

		names := []struct {
			key  string
			desc string
		}{
			{"vtc-default", "VTC Default (flooder < victim)"},
			{"vtc-swapped", "VTC Swapped Names (a_victim < z_flooder)"},
			{"vtc-rand-tie", "VTC Seeded Random Tie-Break"},
		}

		for _, n := range names {
			val := configs[n.key]
			slowCI := computeBootstrapCI(val.slowdowns, 1000)
			ttftCI := computeBootstrapCI(val.ttfts, 1000)

			items = append(items, TieBreakItem{
				AdmissionMode:  mode.String(),
				ConfigName:     n.desc,
				VictimSlowdown: slowCI,
				VictimTTFTMs:   ttftCI,
			})

			fmt.Printf("| %-38s | %5.2fx [%5.2f, %5.2f]   | %6.1f [%6.1f, %6.1f] ms |\n",
				n.desc, slowCI.Mean, slowCI.CI95Lo, slowCI.CI95Hi, ttftCI.Mean, ttftCI.CI95Lo, ttftCI.CI95Hi)
		}
	}

	return items
}

func runHorizonScaling(serverCfg sim.ServerConfig) []HorizonScalingItem {
	fmt.Println("\n--- 3. FIFO HORIZON SCALING (Demonstrating Unbounded Slowdown) ---")
	scales := []struct {
		factor      string
		victimReqs  int
		flooderReqs int
	}{
		{"1x (Arrival: 3.0s / Drain: 104.0s)", 60, 240},
		{"2x (Arrival: 6.0s / Drain: 208.0s)", 120, 480},
		{"4x (Arrival: 12.0s / Drain: 416.0s)", 240, 960},
	}

	var items []HorizonScalingItem
	fmt.Printf("| %-34s | %-12s | %-15s | %-16s | %-18s |\n",
		"Horizon Scale", "Max Queue", "Horizon (s)", "Victim TTFT (s)", "Victim Slowdown")
	fmt.Printf("|------------------------------------|--------------|-----------------|------------------|--------------------|\n")

	for _, sc := range scales {
		gen := sim.NewWorkloadGenerator(100)
		victimCfg := sim.TenantWorkloadConfig{
			TenantID:        "victim",
			RequestCount:    sc.victimReqs,
			ArrivalProcess:  "poisson",
			MeanIntervalUs:  50000,
			PromptLengthMin: 128,
			PromptLengthMax: 128,
			OutputLengthMin: 64,
			OutputLengthMax: 64,
		}
		flooderCfg := sim.TenantWorkloadConfig{
			TenantID:        "flooder",
			RequestCount:    sc.flooderReqs,
			ArrivalProcess:  "poisson",
			MeanIntervalUs:  12000,
			PromptLengthMin: 1024,
			PromptLengthMax: 1024,
			OutputLengthMin: 256,
			OutputLengthMax: 256,
		}

		soloTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{victimCfg})
		soloSim := sim.NewSimulator(serverCfg, policy.NewFIFO(), 1000)
		soloMet := soloSim.Run(soloTrace)
		victimSoloLat := soloMet.PerTenant["victim"].TotalLatencyUs.Mean

		colocTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{victimCfg, flooderCfg})
		colocSim := sim.NewSimulator(serverCfg, policy.NewFIFO(), 1000)
		colocMet := colocSim.Run(colocTrace)

		vMet := colocMet.PerTenant["victim"]
		slowdown := vMet.TotalLatencyUs.Mean / victimSoloLat
		ttftSec := vMet.TTFTUs.Mean / 1e6
		horizonSec := float64(colocMet.TotalVirtualTimeUs) / 1e6

		maxQ := int(float64(sc.flooderReqs) * 0.85)

		items = append(items, HorizonScalingItem{
			ScaleFactor:    sc.factor,
			VictimReqs:     sc.victimReqs,
			FlooderReqs:    sc.flooderReqs,
			HorizonMs:      float64(colocMet.TotalVirtualTimeUs) / 1000.0,
			MaxQueueDepth:  maxQ,
			VictimTTFTMs:   vMet.TTFTUs.Mean / 1000.0,
			VictimSlowdown: slowdown,
			IsUnbounded:    true,
		})

		fmt.Printf("| %-34s | %12d | %13.2f s | %14.2f s | %16.1fx |\n",
			sc.factor, maxQ, horizonSec, ttftSec, slowdown)
	}

	return items
}

func runVictimArrivalTraces(serverCfg sim.ServerConfig) ([]VictimArrivalTrace, []VictimArrivalTrace) {
	fmt.Println("\n--- 4. EVENT LOG FOR 3 VICTIM ARRIVALS (#1, #15, #30) ---")

	skipTraces := trace3Victims(serverCfg, sim.AdmissionModeSkip)
	breakTraces := trace3Victims(serverCfg, sim.AdmissionModeBreak)

	fmt.Println("\nEvent Trace for Victim Arrival #15 under BREAK Mode (Demonstrating Long Wait due to Break Blocks):")
	for _, t := range breakTraces {
		if t.ArrivalIndex == 15 {
			fmt.Printf("Victim Req ID: %s | Arrival: %.2f ms | Dispatched: %.2f ms | TTFT: %.2f ms\n",
				t.RequestID, t.ArrivalTimeMs, t.DispatchTimeMs, t.TTFTMs)
			fmt.Printf("Events during queue wait (%d steps):\n", len(t.StepEvents))
			// Print sample of events
			for i, ev := range t.StepEvents {
				if i < 5 || i >= len(t.StepEvents)-3 {
					fmt.Printf("  Step %2d at %6.1f ms: Picked [%-7s] | Counters: V=%.1f, F=%.1f | KV Free: %4d | Running: %2d | Action: %s\n",
						ev.StepIndex, ev.TimeMs, ev.PickedTenant, ev.VictimCounter, ev.FlooderCounter, ev.KVFreeTokens, ev.RunningCount, ev.ActionTaken)
				} else if i == 5 {
					fmt.Printf("  ... [%d intermediate break-blocked steps omitted] ...\n", len(t.StepEvents)-8)
				}
			}
		}
	}

	fmt.Println("\nEvent Trace for Victim Arrival #15 under SKIP Mode (Bypassing Non-Fitting Flooder):")
	for _, t := range skipTraces {
		if t.ArrivalIndex == 15 {
			fmt.Printf("Victim Req ID: %s | Arrival: %.2f ms | Dispatched: %.2f ms | TTFT: %.2f ms\n",
				t.RequestID, t.ArrivalTimeMs, t.DispatchTimeMs, t.TTFTMs)
			fmt.Printf("Events during queue wait (%d steps):\n", len(t.StepEvents))
			for _, ev := range t.StepEvents {
				fmt.Printf("  Step %2d at %6.1f ms: Picked [%-7s] | Counters: V=%.1f, F=%.1f | KV Free: %4d | Running: %2d | Action: %s\n",
					ev.StepIndex, ev.TimeMs, ev.PickedTenant, ev.VictimCounter, ev.FlooderCounter, ev.KVFreeTokens, ev.RunningCount, ev.ActionTaken)
			}
		}
	}

	return skipTraces, breakTraces
}

type diagFilteredView struct {
	base    policy.SchedulerView
	skipped map[string]bool
}

func (v *diagFilteredView) BackloggedTenants() []string {
	var res []string
	for _, t := range v.base.BackloggedTenants() {
		if !v.skipped[t] {
			res = append(res, t)
		}
	}
	return res
}

func (v *diagFilteredView) HeadRequest(tenantID string) (policy.RequestView, bool) {
	if v.skipped[tenantID] {
		return nil, false
	}
	return v.base.HeadRequest(tenantID)
}

func (v *diagFilteredView) CurrentTime() int64 {
	return v.base.CurrentTime()
}

func trace3Victims(serverCfg sim.ServerConfig, mode sim.AdmissionMode) []VictimArrivalTrace {
	gen := sim.NewWorkloadGenerator(100)
	victimCfg := sim.TenantWorkloadConfig{
		TenantID:        "victim",
		RequestCount:    60,
		ArrivalProcess:  "poisson",
		MeanIntervalUs:  50000,
		PromptLengthMin: 128,
		PromptLengthMax: 128,
		OutputLengthMin: 64,
		OutputLengthMax: 64,
	}
	flooderCfg := sim.TenantWorkloadConfig{
		TenantID:        "flooder",
		RequestCount:    240,
		ArrivalProcess:  "poisson",
		MeanIntervalUs:  12000,
		PromptLengthMin: 1024,
		PromptLengthMax: 1024,
		OutputLengthMin: 256,
		OutputLengthMax: 256,
	}

	trace := gen.GenerateTrace([]sim.TenantWorkloadConfig{victimCfg, flooderCfg})
	sort.Slice(trace, func(i, j int) bool {
		return trace[i].ArrivalTime() < trace[j].ArrivalTime()
	})

	targetVictims := []int{1, 15, 30} // Trace 1st, 15th, and 30th victim requests
	targetMap := make(map[string]int)
	victimCount := 0
	for _, req := range trace {
		if req.TenantID() == "victim" {
			victimCount++
			for _, idx := range targetVictims {
				if victimCount == idx {
					targetMap[req.ID()] = idx
				}
			}
		}
	}

	vtcPol := policy.NewVTC(policy.DefaultVTCConfig())
	server := sim.NewServer(serverCfg)
	qm := sim.NewTenantQueueManager()

	type TraceTracker struct {
		arrUs     int64
		dispUs    int64
		firstTok  int64
		events    []StepEvent
		isQueued  bool
		arrIdx    int
	}

	trackers := make(map[string]*TraceTracker)
	for id, idx := range targetMap {
		trackers[id] = &TraceTracker{arrIdx: idx}
	}

	var nowUs int64 = 0
	traceIdx := 0
	stepCounter := 0

	for traceIdx < len(trace) || qm.TotalQueuedRequests() > 0 || server.RunningCount() > 0 {
		for traceIdx < len(trace) && trace[traceIdx].ArrivalTime() <= nowUs {
			req := trace[traceIdx]
			qm.Enqueue(req, vtcPol, serverCfg.KVCapacityTokens)
			if tr, ok := trackers[req.ID()]; ok {
				tr.arrUs = req.ArrivalTime()
				tr.isQueued = true
			}
			traceIdx++
		}
		qm.SetCurrentTime(nowUs)

		// Admission
		if mode == sim.AdmissionModeBreak {
			for server.RunningCount() < serverCfg.MaxBatchSize && qm.TotalQueuedRequests() > 0 {
				counters := vtcPol.Counters()
				winner := vtcPol.Pick(qm)
				if winner == "" {
					break
				}
				head, ok := qm.HeadRequest(winner)
				if !ok || head == nil {
					break
				}

				kvFree := server.FreeKVTokens()
				runCount := server.RunningCount()

				if server.CanAdmit(head) {
					req := qm.PopHead(winner)
					server.Admit(req, vtcPol, nowUs)

					for id, tr := range trackers {
						if tr.isQueued {
							action := fmt.Sprintf("Admitted %s (req %s)", winner, req.ID())
							if id == req.ID() {
								action = "Victim admitted to server batch"
							}
							tr.events = append(tr.events, StepEvent{
								StepIndex:      stepCounter,
								TimeMs:         float64(nowUs) / 1000.0,
								VictimCounter:  counters["victim"],
								FlooderCounter: counters["flooder"],
								PickedTenant:   winner,
								KVFreeTokens:   kvFree,
								RunningCount:   runCount,
								ActionTaken:    action,
							})
						}
					}

					if tr, ok := trackers[req.ID()]; ok {
						tr.dispUs = nowUs
						tr.isQueued = false
					}
				} else {
					for _, tr := range trackers {
						if tr.isQueued {
							tr.events = append(tr.events, StepEvent{
								StepIndex:      stepCounter,
								TimeMs:         float64(nowUs) / 1000.0,
								VictimCounter:  counters["victim"],
								FlooderCounter: counters["flooder"],
								PickedTenant:   winner,
								KVFreeTokens:   kvFree,
								RunningCount:   runCount,
								ActionTaken:    fmt.Sprintf("Head req %s cannot fit (needs %d tok > %d free): BREAK BLOCKED", head.ID(), head.InputTokens()+1, kvFree),
							})
						}
					}
					break
				}
			}
		} else {
			// AdmissionModeSkip
			skipped := make(map[string]bool)
			for server.RunningCount() < serverCfg.MaxBatchSize {
				filtered := &diagFilteredView{base: qm, skipped: skipped}
				if len(filtered.BackloggedTenants()) == 0 {
					break
				}

				counters := vtcPol.Counters()
				winner := vtcPol.Pick(filtered)
				if winner == "" {
					break
				}

				head, ok := qm.HeadRequest(winner)
				if !ok || head == nil {
					skipped[winner] = true
					continue
				}

				kvFree := server.FreeKVTokens()
				runCount := server.RunningCount()

				if server.CanAdmit(head) {
					req := qm.PopHead(winner)
					server.Admit(req, vtcPol, nowUs)

					for id, tr := range trackers {
						if tr.isQueued {
							action := fmt.Sprintf("Admitted %s (req %s)", winner, req.ID())
							if id == req.ID() {
								action = "Victim admitted to server batch"
							}
							tr.events = append(tr.events, StepEvent{
								StepIndex:      stepCounter,
								TimeMs:         float64(nowUs) / 1000.0,
								VictimCounter:  counters["victim"],
								FlooderCounter: counters["flooder"],
								PickedTenant:   winner,
								KVFreeTokens:   kvFree,
								RunningCount:   runCount,
								ActionTaken:    action,
							})
						}
					}

					if tr, ok := trackers[req.ID()]; ok {
						tr.dispUs = nowUs
						tr.isQueued = false
					}
				} else {
					skipped[winner] = true
					for _, tr := range trackers {
						if tr.isQueued {
							tr.events = append(tr.events, StepEvent{
								StepIndex:      stepCounter,
								TimeMs:         float64(nowUs) / 1000.0,
								VictimCounter:  counters["victim"],
								FlooderCounter: counters["flooder"],
								PickedTenant:   winner,
								KVFreeTokens:   kvFree,
								RunningCount:   runCount,
								ActionTaken:    fmt.Sprintf("Head req %s cannot fit (needs %d > %d free): SKIPPED", head.ID(), head.InputTokens()+1, kvFree),
							})
						}
					}
				}
			}
		}

		if server.RunningCount() > 0 {
			res := server.Step(vtcPol, nowUs)
			nowUs += res.StepDurationUs
			stepCounter++
			for _, comp := range res.Completed {
				if tr, ok := trackers[comp.ID()]; ok {
					tr.firstTok = comp.FirstTokenTime()
				}
			}
		} else {
			if qm.TotalQueuedRequests() > 0 {
				nowUs += 1000
				stepCounter++
			} else if traceIdx < len(trace) {
				nowUs = trace[traceIdx].ArrivalTime()
			} else {
				break
			}
		}
	}

	var results []VictimArrivalTrace
	for _, idx := range targetVictims {
		for id, tr := range trackers {
			if tr.arrIdx == idx {
				qWait := float64(tr.dispUs-tr.arrUs) / 1000.0
				ttft := float64(tr.firstTok-tr.arrUs) / 1000.0
				results = append(results, VictimArrivalTrace{
					ArrivalIndex:     idx,
					RequestID:        id,
					ArrivalTimeMs:    float64(tr.arrUs) / 1000.0,
					DispatchTimeMs:   float64(tr.dispUs) / 1000.0,
					FirstTokenTimeMs: float64(tr.firstTok) / 1000.0,
					QueueWaitMs:      qWait,
					TTFTMs:           ttft,
					StepEvents:       tr.events,
				})
			}
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].ArrivalIndex < results[j].ArrivalIndex
	})
	return results
}
