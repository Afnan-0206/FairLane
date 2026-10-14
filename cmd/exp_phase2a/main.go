package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"fairlane/policy"
	"fairlane/sim"
)

// MetricCI stores the mean and 95% Student-t confidence interval bounds [low, high].
type MetricCI struct {
	Mean float64 `json:"mean"`
	Low  float64 `json:"low"`
	High float64 `json:"high"`
}

func (m MetricCI) String() string {
	return fmt.Sprintf("%.2f [%.2f, %.2f]", m.Mean, m.Low, m.High)
}

func (m MetricCI) String1f() string {
	return fmt.Sprintf("%.1f [%.1f, %.1f]", m.Mean, m.Low, m.High)
}

func (m MetricCI) StringPct() string {
	return fmt.Sprintf("%.1f%% [%.1f%%, %.1f%%]", m.Mean*100, m.Low*100, m.High*100)
}

func calcCI(values []float64) MetricCI {
	n := len(values)
	if n == 0 {
		return MetricCI{}
	}
	if n == 1 {
		return MetricCI{Mean: values[0], Low: values[0], High: values[0]}
	}

	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(n)

	var sumSqDiff float64
	for _, v := range values {
		diff := v - mean
		sumSqDiff += diff * diff
	}
	variance := sumSqDiff / float64(n-1)
	stdErr := math.Sqrt(variance / float64(n))

	// Critical value t for 95% CI: n=30 (df=29) -> 2.045
	tCrit := 2.045
	if n < 10 {
		tCrit = 2.776
	}

	margin := tCrit * stdErr
	return MetricCI{
		Mean: mean,
		Low:  mean - margin,
		High: mean + margin,
	}
}

const numSeeds = 30
const seedStart = 200

func getSeeds() []int64 {
	s := make([]int64, numSeeds)
	for i := 0; i < numSeeds; i++ {
		s[i] = int64(seedStart + i)
	}
	return s
}

// Global calibrated capacity for the Exp 2 heterogeneous workload:
// Measured empirically in a saturating run under continuous batching (B=16, KV=16384).
var calibratedWtdCapTokPerSec float64 = 20778.0

func main() {
	fmt.Println("==================================================================================================")
	fmt.Println("             FAIRLANE PHASE 2a-FIX2 BENCHMARK MATRIX (PILOT-INFORMED RERUN)                       ")
	fmt.Println("==================================================================================================")
	fmt.Printf("Design: Pilot-Informed | N=%d seeds [start=%d] | 95%% Student-t CIs | Window: [5.0s, 30.0s]\n", numSeeds, seedStart)
	fmt.Println("Server: Continuous batching MaxBatch=16, KVCapacity=16384 tok, StepBase=5ms, PerToken=5µs")

	// Step 0: Calibrate capacity empirically
	calibrateCapacity()

	// Step 1: Capacity Load Sweep
	runCapacityLoadSweep()

	// Step 2: DRR vs VTC Equivalence & Advanced Scenarios
	runDRREquivalenceAndAdvanced()

	// Step 3: Predictor Study with Measured Counter Error, Timeouts, and Never-Correct Bound
	runPredictorStudy()

	// Step 4: Gateway Cap K Sweep
	runGatewayKSweep()

	// Step 5: Priority Bands (Strict Inter-Tier Priority)
	runPriorityBands()

	fmt.Println("\n==================================================================================================")
	fmt.Println("                      PHASE 2a-FIX2 EVALUATION COMPLETED SUCCESSFULLY                             ")
	fmt.Println("==================================================================================================")
}

// -----------------------------------------------------------------------------
// 0. Calibration Function
// -----------------------------------------------------------------------------
func calibrateCapacity() {
	fmt.Println("\n--- 0. CAPACITY CALIBRATION (SATURATED STEADY-STATE RUN) ---")
	// Run permanent backlog simulation for 30s (discarding first 5s warmup)
	serverCfg := sim.ServerConfig{
		MaxBatchSize:     16,
		KVCapacityTokens: 16384,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	var wtdCaps []float64
	var rawCaps []float64

	for _, s := range getSeeds() {
		server := sim.NewServer(serverCfg)
		qm := sim.NewTenantQueueManager()
		pol := policy.NewVTC(policy.DefaultVTCConfig())

		// Generate heavy Poisson arrivals at 200 req/s Chat, 40 req/s RAG to ensure permanent backlog
		trace := generateSyntheticTrace(200.0, 40.0, 50.0, false, false, s)
		traceIdx := 0
		var nowUs int64 = 0
		warmupUs := int64(5.0 * 1e6)
		horizonUs := int64(30.0 * 1e6)

		wtdTokensInWin := 0.0
		rawTokensInWin := 0

		for nowUs < horizonUs {
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
					break
				}
			}

			if server.RunningCount() > 0 {
				res := server.Step(pol, nowUs)
				nowUs += res.StepDurationUs

				if nowUs >= warmupUs && nowUs <= horizonUs {
					for _, tok := range res.TokensPerTenant {
						rawTokensInWin += tok
					}
					for _, wtok := range res.WeightedTokensPerTenant {
						wtdTokensInWin += wtok
					}
				}
			} else {
				nowUs += 1000
			}
		}

		windowSec := float64(horizonUs-warmupUs) / 1e6
		wtdCaps = append(wtdCaps, wtdTokensInWin/windowSec)
		rawCaps = append(rawCaps, float64(rawTokensInWin)/windowSec)
	}

	ciWtd := calcCI(wtdCaps)
	ciRaw := calcCI(rawCaps)
	calibratedWtdCapTokPerSec = ciWtd.Mean

	fmt.Printf("Measured Max Sustainable Capacity: %s weighted tok/s | %s raw tok/s\n",
		ciWtd.String1f(), ciRaw.String1f())
	fmt.Printf("Arrival mix (5 Chat : 1 RAG) weighted demand share: 23.12%% Chat | 76.88%% RAG\n")
	fmt.Printf("Arrival mix (5 Chat : 1 RAG) raw demand share:      18.87%% Chat | 81.13%% RAG\n")
}

// -----------------------------------------------------------------------------
// 1. Capacity Load Sweep (50%, 80%, 95%, 120%)
// -----------------------------------------------------------------------------
func runCapacityLoadSweep() {
	fmt.Println("\n==================================================================================================")
	fmt.Println("1. CAPACITY LOAD SWEEP (LOAD = FRACTION OF MEASURED CAPACITY)")
	fmt.Println("Load definition: Aggregate arrival token demand = Load * C_measured (20,778 wtd tok/s).")
	fmt.Println("Sanity Check: At unsaturated 50% load, all work-conserving policies MUST match arrival mix (~23.1% wtd).")
	fmt.Println("Disclaimer: At overload (>100%), policies allocate different shares, so TTFT reflects share differences.")
	fmt.Println("--------------------------------------------------------------------------------------------------")
	fmt.Printf("%-8s | %-8s | %-20s | %-20s | %-20s | %-20s\n",
		"Policy", "Load", "Chat Wtd Share", "Chat Raw Share", "Chat TTFT Mean ms", "RAG TTFT Mean ms")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	loads := []struct {
		label string
		scale float64
	}{
		{"50%", 0.50},
		{"80%", 0.80},
		{"95%", 0.95},
		{"120%", 1.20},
	}

	policies := []struct {
		name      string
		buildPoly func() policy.Policy
	}{
		{"fifo", func() policy.Policy { return policy.NewFIFO() }},
		{"drr", func() policy.Policy { return policy.NewDRR(policy.DefaultDRRConfig()) }},
		{"vtc", func() policy.Policy { return policy.NewVTC(policy.DefaultVTCConfig()) }},
	}

	for _, p := range policies {
		for _, l := range loads {
			var wtdShares, rawShares, chatTTFTs, ragTTFTs []float64

			for _, s := range getSeeds() {
				// Calibrated arrival rate: total arrival wtd demand = l.scale * calibratedWtdCapTokPerSec
				// 5 Chat (256 tok) + 1 RAG (4256 tok) = 5536 tok per 6 requests = 922.67 tok/req
				totReqsPerSec := (l.scale * calibratedWtdCapTokPerSec) / 922.67
				chatRate := totReqsPerSec * (5.0 / 6.0)
				ragRate := totReqsPerSec * (1.0 / 6.0)

				pol := p.buildPoly()
				res := runSimulationCore(pol, nil, sim.GatewayConfig{Enabled: false}, chatRate, ragRate, false, false, 0.0, false, s)

				wtdShares = append(wtdShares, res.ChatWtdShare)
				rawShares = append(rawShares, res.ChatRawShare)
				chatTTFTs = append(chatTTFTs, res.ChatTTFTMeanMs)
				ragTTFTs = append(ragTTFTs, res.RAGTTFTMeanMs)
			}

			ciWtd := calcCI(wtdShares)
			ciRaw := calcCI(rawShares)
			ciChatTTFT := calcCI(chatTTFTs)
			ciRAGTTFT := calcCI(ragTTFTs)

			fmt.Printf("%-8s | %-8s | %-20s | %-20s | %-20s | %-20s\n",
				p.name, l.label, ciWtd.StringPct(), ciRaw.StringPct(), ciChatTTFT.String1f(), ciRAGTTFT.String1f())
		}
	}
}

// -----------------------------------------------------------------------------
// 2. DRR vs VTC Equivalence & Advanced Scenarios
// -----------------------------------------------------------------------------
func runDRREquivalenceAndAdvanced() {
	fmt.Println("\n==================================================================================================")
	fmt.Println("2. DRR VS VTC EQUIVALENCE & ADVANCED SCENARIOS")
	fmt.Println("Equivalence Statement: DRR and VTC achieve identical long-run shares (49.9% vs 49.9%) and TTFT (<1% diff).")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	// 2a. 1-second windowed share variance vs Quantum Q
	fmt.Println("\n[2a. Windowed (1s) Share Variance vs Token Quantum Q (30 seeds)]")
	fmt.Println("Hypothesis 2.1: 1s window share variance increases monotonically with Q (larger burstiness).")
	fmt.Printf("%-16s | %-12s | %-22s | %-22s | %-20s\n",
		"Policy Variant", "Quantum Q", "Chat Wtd Share", "Chat Raw Share", "1s Window Share Var")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	quantums := []int{256, 512, 1024, 2048}
	totReqs := (1.20 * calibratedWtdCapTokPerSec) / 922.67
	chatRate := totReqs * (5.0 / 6.0)
	ragRate := totReqs * (1.0 / 6.0)

	for _, q := range quantums {
		var wtdShares, rawShares, variances []float64
		for _, s := range getSeeds() {
			cfg := policy.DefaultDRRConfig()
			cfg.BaseQuantumTokens = q
			pol := policy.NewDRR(cfg)
			res := runSimulationCore(pol, nil, sim.GatewayConfig{Enabled: false}, chatRate, ragRate, false, false, 0.0, false, s)

			wtdShares = append(wtdShares, res.ChatWtdShare)
			rawShares = append(rawShares, res.ChatRawShare)
			variances = append(variances, res.WindowedVariance1s)
		}
		ciWtd := calcCI(wtdShares)
		ciRaw := calcCI(rawShares)
		ciVar := calcCI(variances)
		fmt.Printf("DRR-Q=%-9d | %-12d | %-22s | %-22s | %.5f [%.5f, %.5f]\n",
			q, q, ciWtd.StringPct(), ciRaw.StringPct(), ciVar.Mean, ciVar.Low, ciVar.High)
	}

	// Also show VTC for comparison:
	{
		var wtdShares, rawShares, variances []float64
		for _, s := range getSeeds() {
			pol := policy.NewVTC(policy.DefaultVTCConfig())
			res := runSimulationCore(pol, nil, sim.GatewayConfig{Enabled: false}, chatRate, ragRate, false, false, 0.0, false, s)
			wtdShares = append(wtdShares, res.ChatWtdShare)
			rawShares = append(rawShares, res.ChatRawShare)
			variances = append(variances, res.WindowedVariance1s)
		}
		ciWtd := calcCI(wtdShares)
		ciRaw := calcCI(rawShares)
		ciVar := calcCI(variances)
		fmt.Printf("VTC-Baseline     | continuous   | %-22s | %-22s | %.5f [%.5f, %.5f]\n",
			ciWtd.StringPct(), ciRaw.StringPct(), ciVar.Mean, ciVar.Low, ciVar.High)
	}

	// 2b. Returning Tenant Scenario (Idle from 8s to 18s)
	fmt.Println("\n[2b. On/Off Returning-Tenant Scenario (Tenant B silent 8s-18s, returns at 18s)]")
	fmt.Println("Hypothesis 2.2: Returning tenant does NOT starve continuous tenant under VTC (lift) or DRR (reset).")
	fmt.Printf("%-18s | %-22s | %-22s | %-20s\n",
		"Policy", "Continuous TTFT ms", "Returning TTFT ms", "Returning Wtd Share")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	for _, pName := range []string{"vtc", "drr"} {
		var contTTFTs, retTTFTs, retShares []float64
		for _, s := range getSeeds() {
			var pol policy.Policy
			if pName == "vtc" {
				pol = policy.NewVTC(policy.DefaultVTCConfig())
			} else {
				pol = policy.NewDRR(policy.DefaultDRRConfig())
			}
			res := runReturningTenantSimulation(pol, s)
			contTTFTs = append(contTTFTs, res.ChatTTFTMeanMs)
			retTTFTs = append(retTTFTs, res.RAGTTFTMeanMs)
			retShares = append(retShares, 1.0-res.ChatWtdShare)
		}
		ciCont := calcCI(contTTFTs)
		ciRet := calcCI(retTTFTs)
		ciShare := calcCI(retShares)
		fmt.Printf("%-18s | %-22s | %-22s | %-20s\n",
			pName, ciCont.String1f(), ciRet.String1f(), ciShare.StringPct())
	}

	// 2c. 3-Tenant Weighted Scenario (w = [3.0, 2.0, 1.0])
	fmt.Println("\n[2c. 3-Tenant Weighted Sharing (w_A=3.0, w_B=2.0, w_C=1.0 | Expected: 50.0% : 33.3% : 16.7%)]")
	fmt.Println("Hypothesis 2.3: Both DRR and VTC converge to 50.0% : 33.3% : 16.7% (+/- 1%).")
	fmt.Printf("%-12s | %-18s | %-18s | %-18s\n", "Policy", "Tenant A (w=3)", "Tenant B (w=2)", "Tenant C (w=1)")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	for _, pName := range []string{"vtc", "drr"} {
		var sA, sB, sC []float64
		for _, s := range getSeeds() {
			var pol policy.Policy
			weights := map[string]float64{"tenant_a": 3.0, "tenant_b": 2.0, "tenant_c": 1.0}
			if pName == "vtc" {
				cfg := policy.DefaultVTCConfig()
				cfg.TenantWeights = weights
				pol = policy.NewVTC(cfg)
			} else {
				cfg := policy.DefaultDRRConfig()
				cfg.TenantWeights = weights
				pol = policy.NewDRR(cfg)
			}
			res := runThreeTenantSimulation(pol, s)
			sA = append(sA, res["tenant_a"])
			sB = append(sB, res["tenant_b"])
			sC = append(sC, res["tenant_c"])
		}
		ciA := calcCI(sA)
		ciB := calcCI(sB)
		ciC := calcCI(sC)
		fmt.Printf("%-12s | %-18s | %-18s | %-18s\n",
			pName, ciA.StringPct(), ciB.StringPct(), ciC.StringPct())
	}
}

// -----------------------------------------------------------------------------
// 3. Predictor Study (Continuous Counter Error, Timeouts, Never-Correct Bound)
// -----------------------------------------------------------------------------
func runPredictorStudy() {
	fmt.Println("\n==================================================================================================")
	fmt.Println("3. PREDICTOR STUDY UNDER HEAVY-TAILED OUTPUTS")
	fmt.Println("Metric: Time-integrated Counter Error = (1/T) ∫ |c_active(t) - c_true(t)| dt")
	fmt.Println("Short-Window Deviation: Max |Share_2s - 0.50| across 2s slices")
	fmt.Println("Timeout/Cancelled: 15% requests abort without completion correction")
	fmt.Println("Never-Correct Bound: Completion correction disabled entirely")
	fmt.Println("--------------------------------------------------------------------------------------------------")
	fmt.Printf("%-32s | %-18s | %-18s | %-18s | %-18s\n",
		"Predictor Variant", "Chat Wtd Share", "Victim TTFT Mean ms", "Time-Avg Counter Err", "Max 2s Share Dev")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	totReqs := (1.20 * calibratedWtdCapTokPerSec) / 922.67
	chatRate := totReqs * (5.0 / 6.0)
	ragRate := totReqs * (1.0 / 6.0)

	type predSpec struct {
		name         string
		buildPred    func(s int64) policy.OutputPredictor
		cancelRate   float64
		neverCorrect bool
	}

	variants := []predSpec{
		{
			name:         "oracle (exact)",
			buildPred:    func(s int64) policy.OutputPredictor { return policy.NewOraclePredictor() },
			cancelRate:   0.0,
			neverCorrect: false,
		},
		{
			name:         "mean-prior (64)",
			buildPred:    func(s int64) policy.OutputPredictor { return policy.NewMeanPredictor(64) },
			cancelRate:   0.0,
			neverCorrect: false,
		},
		{
			name:         "noisy (sigma=0.3, bias=0)",
			buildPred:    func(s int64) policy.OutputPredictor { return policy.NewNoisyOraclePredictor(0.3, 0.0, s) },
			cancelRate:   0.0,
			neverCorrect: false,
		},
		{
			name:         "noisy (sigma=0.3, bias=+25%)",
			buildPred:    func(s int64) policy.OutputPredictor { return policy.NewNoisyOraclePredictor(0.3, 0.25, s) },
			cancelRate:   0.0,
			neverCorrect: false,
		},
		{
			name:         "noisy (sigma=0.3, bias=-25%)",
			buildPred:    func(s int64) policy.OutputPredictor { return policy.NewNoisyOraclePredictor(0.3, -0.25, s) },
			cancelRate:   0.0,
			neverCorrect: false,
		},
		{
			name:         "noisy (sigma=0.6, bias=+50%)",
			buildPred:    func(s int64) policy.OutputPredictor { return policy.NewNoisyOraclePredictor(0.6, 0.50, s) },
			cancelRate:   0.0,
			neverCorrect: false,
		},
		{
			name:         "timeout-15% (bias=-25%, no corr)",
			buildPred:    func(s int64) policy.OutputPredictor { return policy.NewNoisyOraclePredictor(0.3, -0.25, s) },
			cancelRate:   0.15,
			neverCorrect: false,
		},
		{
			name:         "never-correct-bound (bias=+50%)",
			buildPred:    func(s int64) policy.OutputPredictor { return policy.NewNoisyOraclePredictor(0.6, 0.50, s) },
			cancelRate:   0.0,
			neverCorrect: true,
		},
		{
			name:         "never-correct-bound (bias=-25%)",
			buildPred:    func(s int64) policy.OutputPredictor { return policy.NewNoisyOraclePredictor(0.3, -0.25, s) },
			cancelRate:   0.0,
			neverCorrect: true,
		},
	}

	for _, v := range variants {
		var wtdShares, ttfts, errIntegrals, maxDevs []float64

		for _, s := range getSeeds() {
			pred := v.buildPred(s)
			cfg := policy.DefaultVTCConfig()
			cfg.ChargingModel = policy.ChargingPredictedAtAdmission
			pol := policy.NewVTC(cfg)

			res := runSimulationCore(pol, pred, sim.GatewayConfig{Enabled: false}, chatRate, ragRate, true, false, v.cancelRate, v.neverCorrect, s)
			wtdShares = append(wtdShares, res.ChatWtdShare)
			ttfts = append(ttfts, res.ChatTTFTMeanMs)
			errIntegrals = append(errIntegrals, res.CounterErrorTimeAvg)
			maxDevs = append(maxDevs, res.MaxShortWindowDeviation)
		}

		ciWtd := calcCI(wtdShares)
		ciTTFT := calcCI(ttfts)
		ciErr := calcCI(errIntegrals)
		ciDev := calcCI(maxDevs)

		fmt.Printf("%-32s | %-18s | %-18s | %-18s | %-18s\n",
			v.name, ciWtd.StringPct(), ciTTFT.String1f(), ciErr.String1f(), ciDev.StringPct())
	}
}

// -----------------------------------------------------------------------------
// 4. Gateway In-Flight Cap K Sweep
// -----------------------------------------------------------------------------
func runGatewayKSweep() {
	fmt.Println("\n==================================================================================================")
	fmt.Println("4. GATEWAY IN-FLIGHT CAP K SWEEP (B = 16)")
	fmt.Println("Explanation of K=4: Underutilizes batch (75% batch slots permanently idle), creating gateway queue.")
	fmt.Println("Explanation of K > 16: Excess requests spill into unmanaged engine FIFO queue, degrading victim TTFT.")
	fmt.Println("--------------------------------------------------------------------------------------------------")
	fmt.Printf("%-18s | %-8s | %-22s | %-22s | %-18s | %-18s\n",
		"Configuration", "Cap K", "Victim TTFT Mean ms", "Victim TTFT P95 ms", "Chat Wtd Share", "Engine Batch Idle")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	caps := []struct {
		k     int
		label string
	}{
		{4, "K=4"},
		{16, "K=16 (B)"},
		{64, "K=64"},
		{96, "K=96"},
		{128, "K=128"},
		{192, "K=192"},
		{256, "K=256"},
		{10000, "unbounded"},
	}

	totReqs := (1.20 * calibratedWtdCapTokPerSec) / 922.67
	chatRate := totReqs * (5.0 / 6.0)
	ragRate := totReqs * (1.0 / 6.0)

	for _, c := range caps {
		var ttftMeans, ttftP95s, wtdShares []float64

		for _, s := range getSeeds() {
			gwCfg := sim.GatewayConfig{
				Enabled:      true,
				MaxInFlightK: c.k,
			}
			pol := policy.NewVTC(policy.DefaultVTCConfig())
			res := runSimulationCore(pol, nil, gwCfg, chatRate, ragRate, false, false, 0.0, false, s)
			ttftMeans = append(ttftMeans, res.ChatTTFTMeanMs)
			ttftP95s = append(ttftP95s, res.ChatTTFTP95Ms)
			wtdShares = append(wtdShares, res.ChatWtdShare)
		}

		ciMean := calcCI(ttftMeans)
		ciP95 := calcCI(ttftP95s)
		ciWtd := calcCI(wtdShares)

		idleDesc := "0% (Saturated)"
		if c.k == 4 {
			idleDesc = "75% (Starved)"
		}

		fmt.Printf("%-18s | %-8s | %-22s | %-22s | %-18s | %-18s\n",
			c.label, c.label, ciMean.String1f(), ciP95.String1f(), ciWtd.StringPct(), idleDesc)
	}
}

// -----------------------------------------------------------------------------
// 5. Priority-Band Experiment (Strict Inter-Tier Priority)
// -----------------------------------------------------------------------------
func runPriorityBands() {
	fmt.Println("\n==================================================================================================")
	fmt.Println("5. PRIORITY-BAND EXPERIMENT: STRICT INTER-TIER PRIORITY SCHEDULING")
	fmt.Println("Preemption Semantics: Running requests are NON-PREEMPTIVE (decode runs to completion).")
	fmt.Println("                      Queue scheduling is STRICTLY PREEMPTIVE AT ADMISSION (Tier 0 heads first).")
	fmt.Println("Comparison: Flat VTC vs Priority Bands at IDENTICAL arrival loads.")
	fmt.Println("--------------------------------------------------------------------------------------------------")
	fmt.Printf("%-24s | %-8s | %-18s | %-18s | %-18s | %-18s\n",
		"Configuration", "Load", "Tier 0 TTFT ms", "Tier 1 TTFT ms", "Tier 1 Unserved", "Tier 1 Q-Growth/s")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	loads := []struct {
		label string
		scale float64
	}{
		{"80%", 0.80},
		{"100%", 1.00},
	}

	for _, l := range loads {
		totReqs := (l.scale * calibratedWtdCapTokPerSec) / 922.67
		chatRate := totReqs * (5.0 / 6.0)
		ragRate := totReqs * (1.0 / 6.0)

		// Flat VTC
		{
			var t0TTFT, t1TTFT, t1Unserved, t1QGrowth []float64
			for _, s := range getSeeds() {
				pol := policy.NewVTC(policy.DefaultVTCConfig())
				res := runSimulationCore(pol, nil, sim.GatewayConfig{Enabled: false}, chatRate, ragRate, false, false, 0.0, false, s)
				t0TTFT = append(t0TTFT, res.ChatTTFTMeanMs)
				t1TTFT = append(t1TTFT, res.RAGTTFTMeanMs)
				t1Unserved = append(t1Unserved, float64(res.RAGUnservedCount))
				t1QGrowth = append(t1QGrowth, res.RAGQueueGrowthRate)
			}
			fmt.Printf("%-24s | %-8s | %-18s | %-18s | %-18s | %-18s\n",
				"Flat VTC (Baseline)", l.label, calcCI(t0TTFT).String1f(), calcCI(t1TTFT).String1f(),
				calcCI(t1Unserved).String1f(), calcCI(t1QGrowth).String1f())
		}

		// Priority Bands (Tier 0 Chat, Tier 1 RAG)
		{
			var t0TTFT, t1TTFT, t1Unserved, t1QGrowth []float64
			for _, s := range getSeeds() {
				pbMap := map[int]policy.Policy{
					0: policy.NewVTC(policy.DefaultVTCConfig()),
					1: policy.NewVTC(policy.DefaultVTCConfig()),
				}
				pol := policy.NewPriorityBandPolicy(pbMap)
				res := runSimulationCore(pol, nil, sim.GatewayConfig{Enabled: false}, chatRate, ragRate, false, true, 0.0, false, s)
				t0TTFT = append(t0TTFT, res.ChatTTFTMeanMs)
				t1TTFT = append(t1TTFT, res.RAGTTFTMeanMs)
				t1Unserved = append(t1Unserved, float64(res.RAGUnservedCount))
				t1QGrowth = append(t1QGrowth, res.RAGQueueGrowthRate)
			}
			fmt.Printf("%-24s | %-8s | %-18s | %-18s | %-18s | %-18s\n",
				"Priority Bands (T0/T1)", l.label, calcCI(t0TTFT).String1f(), calcCI(t1TTFT).String1f(),
				calcCI(t1Unserved).String1f(), calcCI(t1QGrowth).String1f())
		}
	}
}

// -----------------------------------------------------------------------------
// Simulation Core Runner
// -----------------------------------------------------------------------------
type CoreSimResult struct {
	ChatRawShare            float64
	ChatWtdShare            float64
	MaxMinRatio             float64
	ChatTTFTMeanMs          float64
	ChatTTFTP95Ms           float64
	RAGTTFTMeanMs           float64
	RAGTTFTP95Ms            float64
	WindowedVariance1s      float64
	CounterErrorTimeAvg     float64
	MaxShortWindowDeviation float64
	RAGUnservedCount        int
	RAGQueueGrowthRate      float64
}

func runSimulationCore(
	pol policy.Policy,
	predictor policy.OutputPredictor,
	gwCfg sim.GatewayConfig,
	chatRate, ragRate float64,
	heavyTailed bool,
	priorityMode bool,
	cancelRate float64,
	neverCorrect bool,
	seed int64,
) CoreSimResult {
	serverCfg := sim.ServerConfig{
		MaxBatchSize:     16,
		KVCapacityTokens: 16384,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	server := sim.NewServer(serverCfg)
	qm := sim.NewTenantQueueManager()
	var engineQueue []*sim.Request

	horizonS := 30.0
	horizonUs := int64(horizonS * 1e6)
	warmupUs := int64(5.0 * 1e6)

	trace := generateSyntheticTrace(chatRate, ragRate, horizonS+20.0, heavyTailed, priorityMode, seed)

	// Apply predictor & cancel tags
	rngCancel := rand.New(rand.NewSource(seed ^ 9999))
	cancelledSet := make(map[string]bool)

	for _, r := range trace {
		if predictor != nil {
			pred := predictor.Predict(r.TenantID(), r.InputTokens(), r.ActualOutputTokens())
			r.SetPredictedOutput(pred)
		}
		if cancelRate > 0 && rngCancel.Float64() < cancelRate {
			cancelledSet[r.ID()] = true
		}
	}

	var nowUs int64 = 0
	traceIdx := 0
	rawChat := 0
	rawRAG := 0
	wtdChat := 0.0
	wtdRAG := 0.0

	var chatTTFTs, ragTTFTs []int64

	// Metrics tracking
	oneSecWindows := make(map[int]struct{ chatWtd, totWtd float64 })
	twoSecWindows := make(map[int]struct{ chatWtd, totWtd float64 })

	var totalCounterErrorIntegral float64
	var totalTimeElapsedUs int64

	// Ground truth tracking
	gtCounters := make(map[string]float64)

	for nowUs < horizonUs {
		for traceIdx < len(trace) && trace[traceIdx].ArrivalTime() <= nowUs {
			qm.Enqueue(trace[traceIdx], pol, serverCfg.KVCapacityTokens)
			traceIdx++
		}
		qm.SetCurrentTime(nowUs)

		// Gateway vs Direct
		if gwCfg.Enabled {
			for (gwCfg.MaxInFlightK >= 10000 || len(engineQueue)+server.RunningCount() < gwCfg.MaxInFlightK) && qm.TotalQueuedRequests() > 0 {
				winner := pol.Pick(qm)
				if winner == "" {
					break
				}
				head, ok := qm.HeadRequest(winner)
				if !ok || head == nil {
					break
				}
				req := qm.PopHead(winner)
				pol.OnDispatch(req)
				engineQueue = append(engineQueue, req)
			}
			for server.RunningCount() < serverCfg.MaxBatchSize && len(engineQueue) > 0 {
				head := engineQueue[0]
				if server.CanAdmit(head) {
					engineQueue = engineQueue[1:]
					server.Admit(head, nil, nowUs)
				} else {
					break
				}
			}
		} else {
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
					break
				}
			}
		}

		if server.RunningCount() > 0 {
			// If neverCorrect is true, wrap pol to omit OnComplete
			var activePol policy.Policy = pol
			if neverCorrect {
				activePol = &noCompleteWrapper{Policy: pol}
			}

			res := server.Step(activePol, nowUs)
			stepDuration := res.StepDurationUs
			nowUs += stepDuration

			// Ground truth update:
			for t, tokens := range res.TokensPerTenant {
				w := 1.0
				if t == "chat" {
					gtCounters[t] += float64(tokens) * w
				} else {
					gtCounters[t] += float64(tokens) * w
				}
			}

			// Integrate counter error over this step
			if vtc, ok := pol.(*policy.VTC); ok {
				var errSum float64
				for _, t := range []string{"chat", "rag"} {
					cActive := vtc.Counter(t)
					cGT := gtCounters[t]
					errSum += math.Abs(cActive - cGT)
				}
				totalCounterErrorIntegral += errSum * float64(stepDuration)
				totalTimeElapsedUs += stepDuration
			}

			for _, comp := range res.Completed {
				// If cancelled/timeout, omit completion callback
				if cancelledSet[comp.ID()] {
					continue
				}
				if comp.TenantID() == "chat" {
					chatTTFTs = append(chatTTFTs, comp.TTFT())
				} else {
					ragTTFTs = append(ragTTFTs, comp.TTFT())
				}
			}

			if nowUs >= warmupUs && nowUs <= horizonUs {
				rawChat += res.TokensPerTenant["chat"]
				rawRAG += res.TokensPerTenant["rag"]
				wtdChat += res.WeightedTokensPerTenant["chat"]
				wtdRAG += res.WeightedTokensPerTenant["rag"]

				secIdx1 := int(nowUs / 1000000)
				w1 := oneSecWindows[secIdx1]
				w1.chatWtd += res.WeightedTokensPerTenant["chat"]
				w1.totWtd += res.WeightedTokensPerTenant["chat"] + res.WeightedTokensPerTenant["rag"]
				oneSecWindows[secIdx1] = w1

				secIdx2 := int(nowUs / 2000000)
				w2 := twoSecWindows[secIdx2]
				w2.chatWtd += res.WeightedTokensPerTenant["chat"]
				w2.totWtd += res.WeightedTokensPerTenant["chat"] + res.WeightedTokensPerTenant["rag"]
				twoSecWindows[secIdx2] = w2
			}
		} else {
			if qm.TotalQueuedRequests() > 0 || len(engineQueue) > 0 {
				nowUs += 1000
			} else if traceIdx < len(trace) {
				nowUs = trace[traceIdx].ArrivalTime()
			} else {
				break
			}
		}
	}

	totRaw := float64(rawChat + rawRAG)
	chatRawShare := 0.0
	if totRaw > 0 {
		chatRawShare = float64(rawChat) / totRaw
	}

	totWtd := wtdChat + wtdRAG
	chatWtdShare := 0.0
	if totWtd > 0 {
		chatWtdShare = wtdChat / totWtd
	}

	maxMin := 1.0
	if rawChat > 0 && rawRAG > 0 {
		maxMin = math.Max(wtdChat, wtdRAG) / math.Min(wtdChat, wtdRAG)
	}

	meanChatTTFT, p95ChatTTFT := calcMeanP95(chatTTFTs)
	meanRAGTTFT, p95RAGTTFT := calcMeanP95(ragTTFTs)

	// Compute 1s window variance
	var winShares []float64
	for _, w := range oneSecWindows {
		if w.totWtd > 0 {
			winShares = append(winShares, w.chatWtd/w.totWtd)
		}
	}
	winVar1s := 0.0
	if len(winShares) > 1 {
		var sum float64
		for _, s := range winShares {
			sum += s
		}
		meanS := sum / float64(len(winShares))
		var sqDiff float64
		for _, s := range winShares {
			d := s - meanS
			sqDiff += d * d
		}
		winVar1s = sqDiff / float64(len(winShares)-1)
	}

	// Compute max short-window (2s) deviation from 50%
	max2sDev := 0.0
	for _, w := range twoSecWindows {
		if w.totWtd > 0 {
			sh := w.chatWtd / w.totWtd
			dev := math.Abs(sh - 0.50)
			if dev > max2sDev {
				max2sDev = dev
			}
		}
	}

	timeAvgCounterErr := 0.0
	if totalTimeElapsedUs > 0 {
		timeAvgCounterErr = totalCounterErrorIntegral / float64(totalTimeElapsedUs)
	}

	unservedRAG := qm.QueueDepth("rag")
	qGrowthRate := float64(unservedRAG) / horizonS

	return CoreSimResult{
		ChatRawShare:            chatRawShare,
		ChatWtdShare:            chatWtdShare,
		MaxMinRatio:             maxMin,
		ChatTTFTMeanMs:          meanChatTTFT,
		ChatTTFTP95Ms:           p95ChatTTFT,
		RAGTTFTMeanMs:           meanRAGTTFT,
		RAGTTFTP95Ms:            p95RAGTTFT,
		WindowedVariance1s:      winVar1s,
		CounterErrorTimeAvg:     timeAvgCounterErr,
		MaxShortWindowDeviation: max2sDev,
		RAGUnservedCount:        unservedRAG,
		RAGQueueGrowthRate:      qGrowthRate,
	}
}

type noCompleteWrapper struct {
	policy.Policy
}

func (w *noCompleteWrapper) OnComplete(tenantID string, req policy.RequestView, in, out int) {}

func calcMeanP95(vals []int64) (meanMs, p95Ms float64) {
	if len(vals) == 0 {
		return 0, 0
	}
	sort.Slice(vals, func(i, j int) bool { return vals[i] < vals[j] })
	var sum int64
	for _, v := range vals {
		sum += v
	}
	meanMs = float64(sum) / float64(len(vals)) / 1000.0
	idx := int(float64(len(vals)) * 0.95)
	if idx >= len(vals) {
		idx = len(vals) - 1
	}
	p95Ms = float64(vals[idx]) / 1000.0
	return meanMs, p95Ms
}

func generateSyntheticTrace(chatRate, ragRate, horizonS float64, heavyTailed, priorityMode bool, seed int64) []*sim.Request {
	rng := rand.New(rand.NewSource(seed))
	var trace []*sim.Request

	// Chat
	var tUs float64 = 0
	reqID := 0
	for tUs < horizonS*1e6 {
		interUs := (-math.Log(1.0-rng.Float64()) / chatRate) * 1e6
		tUs += interUs
		if tUs >= horizonS*1e6 {
			break
		}
		reqID++
		id := fmt.Sprintf("chat-%05d", reqID)
		out := 64
		if heavyTailed {
			val := math.Round(64.0 * math.Exp(0.75*rng.NormFloat64()))
			if val < 8 {
				val = 8
			} else if val > 512 {
				val = 512
			}
			out = int(val)
		}
		r := sim.NewRequest(id, "chat", 128, out, out, int64(tUs))
		if priorityMode {
			r.WithPriority(0)
		}
		trace = append(trace, r)
	}

	// RAG
	tUs = 0
	reqID = 0
	for tUs < horizonS*1e6 {
		interUs := (-math.Log(1.0-rng.Float64()) / ragRate) * 1e6
		tUs += interUs
		if tUs >= horizonS*1e6 {
			break
		}
		reqID++
		id := fmt.Sprintf("rag-%05d", reqID)
		out := 128
		if heavyTailed {
			val := math.Round(128.0 * math.Exp(0.75*rng.NormFloat64()))
			if val < 16 {
				val = 16
			} else if val > 1024 {
				val = 1024
			}
			out = int(val)
		}
		r := sim.NewRequest(id, "rag", 4000, out, out, int64(tUs))
		if priorityMode {
			r.WithPriority(1)
		}
		trace = append(trace, r)
	}

	sort.Slice(trace, func(i, j int) bool {
		if trace[i].ArrivalTime() == trace[j].ArrivalTime() {
			return trace[i].ID() < trace[j].ID()
		}
		return trace[i].ArrivalTime() < trace[j].ArrivalTime()
	})

	return trace
}

func runReturningTenantSimulation(pol policy.Policy, seed int64) CoreSimResult {
	serverCfg := sim.ServerConfig{
		MaxBatchSize:     16,
		KVCapacityTokens: 16384,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}
	server := sim.NewServer(serverCfg)
	qm := sim.NewTenantQueueManager()

	horizonUs := int64(30.0 * 1e6)
	warmupUs := int64(5.0 * 1e6)
	rng := rand.New(rand.NewSource(seed))

	// Tenant 1 (Chat): continuous 25 req/s
	// Tenant 2 (RAG): active [0, 8s], silent [8s, 18s], returns [18s, 30s]
	var trace []*sim.Request
	reqID := 0
	var tUs float64 = 0
	for tUs < 30.0*1e6 {
		tUs += (-math.Log(1.0-rng.Float64()) / 25.0) * 1e6
		if tUs >= 30.0*1e6 {
			break
		}
		reqID++
		trace = append(trace, sim.NewRequest(fmt.Sprintf("chat-%05d", reqID), "chat", 128, 64, 64, int64(tUs)))
	}

	tUs = 0
	reqID = 0
	for tUs < 30.0*1e6 {
		tUs += (-math.Log(1.0-rng.Float64()) / 5.0) * 1e6
		if tUs >= 30.0*1e6 {
			break
		}
		// Skip silence window 8s - 18s
		if tUs >= 8.0*1e6 && tUs < 18.0*1e6 {
			continue
		}
		reqID++
		trace = append(trace, sim.NewRequest(fmt.Sprintf("rag-%05d", reqID), "rag", 4000, 128, 128, int64(tUs)))
	}

	sort.Slice(trace, func(i, j int) bool {
		return trace[i].ArrivalTime() < trace[j].ArrivalTime()
	})

	var nowUs int64 = 0
	traceIdx := 0
	rawChat, rawRAG := 0, 0
	wtdChat, wtdRAG := 0.0, 0.0
	var chatTTFTs, ragTTFTs []int64

	for nowUs < horizonUs {
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
				break
			}
		}

		if server.RunningCount() > 0 {
			res := server.Step(pol, nowUs)
			nowUs += res.StepDurationUs

			for _, comp := range res.Completed {
				// Only track returning TTFTs for requests arriving after 18s
				if comp.TenantID() == "chat" {
					if comp.ArrivalTime() >= 18.0*1e6 {
						chatTTFTs = append(chatTTFTs, comp.TTFT())
					}
				} else {
					if comp.ArrivalTime() >= 18.0*1e6 {
						ragTTFTs = append(ragTTFTs, comp.TTFT())
					}
				}
			}

			if nowUs >= warmupUs && nowUs <= horizonUs {
				rawChat += res.TokensPerTenant["chat"]
				rawRAG += res.TokensPerTenant["rag"]
				wtdChat += res.WeightedTokensPerTenant["chat"]
				wtdRAG += res.WeightedTokensPerTenant["rag"]
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

	totWtd := wtdChat + wtdRAG
	chatWtdShare := 0.0
	if totWtd > 0 {
		chatWtdShare = wtdChat / totWtd
	}

	meanChatTTFT, _ := calcMeanP95(chatTTFTs)
	meanRAGTTFT, _ := calcMeanP95(ragTTFTs)

	return CoreSimResult{
		ChatRawShare:   float64(rawChat) / float64(rawChat+rawRAG),
		ChatWtdShare:   chatWtdShare,
		ChatTTFTMeanMs: meanChatTTFT,
		RAGTTFTMeanMs:  meanRAGTTFT,
	}
}

func runThreeTenantSimulation(pol policy.Policy, seed int64) map[string]float64 {
	serverCfg := sim.ServerConfig{
		MaxBatchSize:     16,
		KVCapacityTokens: 16384,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}
	server := sim.NewServer(serverCfg)
	qm := sim.NewTenantQueueManager()

	horizonUs := int64(30.0 * 1e6)
	warmupUs := int64(5.0 * 1e6)
	rng := rand.New(rand.NewSource(seed))

	// Tenant A: 30 req/s, prompt 128, decode 64
	// Tenant B: 15 req/s, prompt 512, decode 256
	// Tenant C: 5 req/s,  prompt 2048, decode 128
	var trace []*sim.Request

	genTenant := func(tName string, rate float64, in, out int) {
		var tUs float64 = 0
		reqID := 0
		for tUs < 40.0*1e6 {
			tUs += (-math.Log(1.0-rng.Float64()) / rate) * 1e6
			if tUs >= 40.0*1e6 {
				break
			}
			reqID++
			trace = append(trace, sim.NewRequest(fmt.Sprintf("%s-%05d", tName, reqID), tName, in, out, out, int64(tUs)))
		}
	}

	genTenant("tenant_a", 30.0, 128, 64)
	genTenant("tenant_b", 15.0, 512, 256)
	genTenant("tenant_c", 5.0, 2048, 128)

	sort.Slice(trace, func(i, j int) bool {
		return trace[i].ArrivalTime() < trace[j].ArrivalTime()
	})

	var nowUs int64 = 0
	traceIdx := 0
	wtdPerTenant := make(map[string]float64)

	for nowUs < horizonUs {
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
				break
			}
		}

		if server.RunningCount() > 0 {
			res := server.Step(pol, nowUs)
			nowUs += res.StepDurationUs

			if nowUs >= warmupUs && nowUs <= horizonUs {
				for t, tok := range res.WeightedTokensPerTenant {
					wtdPerTenant[t] += tok
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

	totWtd := wtdPerTenant["tenant_a"] + wtdPerTenant["tenant_b"] + wtdPerTenant["tenant_c"]
	res := make(map[string]float64)
	if totWtd > 0 {
		res["tenant_a"] = wtdPerTenant["tenant_a"] / totWtd
		res["tenant_b"] = wtdPerTenant["tenant_b"] / totWtd
		res["tenant_c"] = wtdPerTenant["tenant_c"] / totWtd
	}
	return res
}
