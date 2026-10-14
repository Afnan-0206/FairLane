package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"

	"fairlane/policy"
	"fairlane/sim"
)

// MetricCI stores the mean and 95% confidence interval bounds [low, high].
type MetricCI struct {
	Mean float64 `json:"mean"`
	Low  float64 `json:"low"`
	High float64 `json:"high"`
}

func (m MetricCI) String() string {
	return fmt.Sprintf("%.1f [%.1f, %.1f]", m.Mean, m.Low, m.High)
}

func (m MetricCI) StringPct() string {
	return fmt.Sprintf("%.1f%% [%.1f%%, %.1f%%]", m.Mean*100, m.Low*100, m.High*100)
}

// calcCI computes mean and 95% Student-t confidence interval across multiple seeds.
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

	// Critical value t for 95% CI:
	// n=5 (df=4) -> 2.776
	// n=30 (df=29) -> 2.045
	// general default:
	tCrit := 2.776
	if n > 10 {
		tCrit = 2.045
	}

	margin := tCrit * stdErr
	return MetricCI{
		Mean: mean,
		Low:  mean - margin,
		High: mean + margin,
	}
}

type ExperimentResult struct {
	Name            string   `json:"name"`
	Category        string   `json:"category"`
	Parameter       string   `json:"parameter"`
	ChatWtdShare    MetricCI `json:"chat_wtd_share"`
	ChatRawShare    MetricCI `json:"chat_raw_share"`
	MaxMinRatio     MetricCI `json:"max_min_ratio"`
	ChatTTFTMeanMs  MetricCI `json:"chat_ttft_mean_ms"`
	ChatTTFTP95Ms   MetricCI `json:"chat_ttft_p95_ms"`
	RAGTTFTMeanMs   MetricCI `json:"rag_ttft_mean_ms"`
	RAGTTFTP95Ms    MetricCI `json:"rag_ttft_p95_ms"`
	PredictorMAE    MetricCI `json:"predictor_mae_tokens"`
	CounterDrift    MetricCI `json:"counter_drift_tokens"`
	SeedsCount      int      `json:"seeds_count"`
}

type SingleRunResult struct {
	ChatRawShare   float64
	ChatWtdShare   float64
	MaxMinRatio    float64
	ChatTTFTMeanMs float64
	ChatTTFTP95Ms  float64
	RAGTTFTMeanMs  float64
	RAGTTFTP95Ms   float64
	PredictorMAE   float64
	CounterDrift   float64
}

var seeds = []int64{12345, 23456, 34567, 45678, 56789}

func aggregateRuns(name, category, param string, runs []SingleRunResult) ExperimentResult {
	wtdShares := make([]float64, len(runs))
	rawShares := make([]float64, len(runs))
	maxMins := make([]float64, len(runs))
	chatTTFTMeans := make([]float64, len(runs))
	chatTTFTP95s := make([]float64, len(runs))
	ragTTFTMeans := make([]float64, len(runs))
	ragTTFTP95s := make([]float64, len(runs))
	maes := make([]float64, len(runs))
	drifts := make([]float64, len(runs))

	for i, r := range runs {
		wtdShares[i] = r.ChatWtdShare
		rawShares[i] = r.ChatRawShare
		maxMins[i] = r.MaxMinRatio
		chatTTFTMeans[i] = r.ChatTTFTMeanMs
		chatTTFTP95s[i] = r.ChatTTFTP95Ms
		ragTTFTMeans[i] = r.RAGTTFTMeanMs
		ragTTFTP95s[i] = r.RAGTTFTP95Ms
		maes[i] = r.PredictorMAE
		drifts[i] = r.CounterDrift
	}

	return ExperimentResult{
		Name:           name,
		Category:       category,
		Parameter:      param,
		ChatWtdShare:   calcCI(wtdShares),
		ChatRawShare:   calcCI(rawShares),
		MaxMinRatio:    calcCI(maxMins),
		ChatTTFTMeanMs: calcCI(chatTTFTMeans),
		ChatTTFTP95Ms:  calcCI(chatTTFTP95s),
		RAGTTFTMeanMs:  calcCI(ragTTFTMeans),
		RAGTTFTP95Ms:   calcCI(ragTTFTP95s),
		PredictorMAE:   calcCI(maes),
		CounterDrift:   calcCI(drifts),
		SeedsCount:     len(runs),
	}
}

func main() {
	fmt.Println("==================================================================================================")
	fmt.Println("                   FAIRLANE PHASE 2a BENCHMARK MATRIX EVALUATION (FIX GATE)                       ")
	fmt.Println("==================================================================================================")
	fmt.Printf("Evaluation parameters: N=%d seeds %v, 95%% Student-t CIs, Window: [5.0s, 30.0s]\n", len(seeds), seeds)
	fmt.Println("Continuous batching server: MaxBatch=16, KVCapacity=16384 tok, StepBase=5ms, PerToken=5µs")

	var allResults []ExperimentResult

	// -------------------------------------------------------------------------
	// 1. DRR Token Quantum & Weight Sweep
	// -------------------------------------------------------------------------
	fmt.Println("\n==================================================================================================")
	fmt.Println("1. DEFICIT ROUND ROBIN (DRR) TOKEN QUANTUM & WEIGHT EVALUATION")
	fmt.Println("Invariant: DRR charges weighted tokens (w_in*in + w_out*pred_out) corrected on completion.")
	fmt.Println("Hypothesis: Quantum determines burstiness/granularity, NOT long-run share (~50.0% weighted share).")
	fmt.Println("--------------------------------------------------------------------------------------------------")
	fmt.Printf("%-18s | %-12s | %-24s | %-24s | %-18s | %-18s\n",
		"Configuration", "Quantum", "Chat Wtd Share (95% CI)", "Chat Raw Share (95% CI)", "Max-Min (95% CI)", "Chat TTFT ms (CI)")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	quantums := []int{256, 512, 1024, 2048}
	for _, q := range quantums {
		var runs []SingleRunResult
		for _, seed := range seeds {
			cfg := policy.DefaultDRRConfig()
			cfg.BaseQuantumTokens = q
			cfg.InputWeight = 1.0
			cfg.OutputWeight = 2.0
			cfg.UsePredictedCost = true
			pol := policy.NewDRR(cfg)

			res := runSimulation(pol, nil, sim.GatewayConfig{Enabled: false}, 1.0, false, false, seed)
			runs = append(runs, res)
		}
		agg := aggregateRuns(fmt.Sprintf("drr-q%d", q), "drr_quantum", fmt.Sprintf("Q=%d", q), runs)
		allResults = append(allResults, agg)
		fmt.Printf("%-18s | %-12d | %-24s | %-24s | %-18s | %-18s\n",
			agg.Name, q, agg.ChatWtdShare.StringPct(), agg.ChatRawShare.StringPct(), agg.MaxMinRatio.String(), agg.ChatTTFTMeanMs.String())
	}

	// Weighted DRR (Chat w=2.0, RAG w=1.0, Q=512)
	{
		var runs []SingleRunResult
		for _, seed := range seeds {
			cfg := policy.DefaultDRRConfig()
			cfg.BaseQuantumTokens = 512
			cfg.InputWeight = 1.0
			cfg.OutputWeight = 2.0
			cfg.TenantWeights = map[string]float64{"chat": 2.0, "rag": 1.0}
			cfg.UsePredictedCost = true
			pol := policy.NewDRR(cfg)

			res := runSimulation(pol, nil, sim.GatewayConfig{Enabled: false}, 1.0, false, false, seed)
			runs = append(runs, res)
		}
		agg := aggregateRuns("drr-weighted-2x", "drr_weighted", "w_chat=2,w_rag=1", runs)
		allResults = append(allResults, agg)
		fmt.Printf("%-18s | %-12s | %-24s | %-24s | %-18s | %-18s\n",
			agg.Name, "512 (w=2:1)", agg.ChatWtdShare.StringPct(), agg.ChatRawShare.StringPct(), agg.MaxMinRatio.String(), agg.ChatTTFTMeanMs.String())
	}

	// -------------------------------------------------------------------------
	// 2. Predictor Study Under Heavy-Tailed Output Lengths
	// -------------------------------------------------------------------------
	fmt.Println("\n==================================================================================================")
	fmt.Println("2. PREDICTOR STUDY UNDER HEAVY-TAILED OUTPUT LENGTHS (VTC Predicted-Admission)")
	fmt.Println("Workload: Log-Normal outputs (Chat med=64 tok, sigma=0.75; RAG med=128 tok, sigma=0.75)")
	fmt.Println("Charging: w_in*in + w_out*predicted_out at admission, corrected by w_out*(actual - pred) on completion.")
	fmt.Println("--------------------------------------------------------------------------------------------------")
	fmt.Printf("%-32s | %-18s | %-18s | %-20s | %-16s | %-16s\n",
		"Predictor Variant", "Chat Wtd Share", "Chat Raw Share", "Victim TTFT Mean ms", "Predictor MAE", "Counter Drift")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	type predSpec struct {
		name      string
		param     string
		buildPred func(seed int64) policy.OutputPredictor
	}

	predVariants := []predSpec{
		{
			name:      "pred-oracle",
			param:     "oracle",
			buildPred: func(s int64) policy.OutputPredictor { return policy.NewOraclePredictor() },
		},
		{
			name:      "pred-mean-prior-64",
			param:     "mean(64)",
			buildPred: func(s int64) policy.OutputPredictor { return policy.NewMeanPredictor(64) },
		},
		{
			name:  "pred-noisy (sigma=0.3, bias=0)",
			param: "sigma=0.3,bias=0.0",
			buildPred: func(s int64) policy.OutputPredictor {
				return policy.NewNoisyOraclePredictor(0.3, 0.0, s)
			},
		},
		{
			name:  "pred-noisy (sigma=0.3, bias=+25%)",
			param: "sigma=0.3,bias=+0.25",
			buildPred: func(s int64) policy.OutputPredictor {
				return policy.NewNoisyOraclePredictor(0.3, 0.25, s)
			},
		},
		{
			name:  "pred-noisy (sigma=0.3, bias=-25%)",
			param: "sigma=0.3,bias=-0.25",
			buildPred: func(s int64) policy.OutputPredictor {
				return policy.NewNoisyOraclePredictor(0.3, -0.25, s)
			},
		},
		{
			name:  "pred-noisy (sigma=0.6, bias=0)",
			param: "sigma=0.6,bias=0.0",
			buildPred: func(s int64) policy.OutputPredictor {
				return policy.NewNoisyOraclePredictor(0.6, 0.0, s)
			},
		},
		{
			name:  "pred-noisy (sigma=0.6, bias=+50%)",
			param: "sigma=0.6,bias=+0.50",
			buildPred: func(s int64) policy.OutputPredictor {
				return policy.NewNoisyOraclePredictor(0.6, 0.50, s)
			},
		},
	}

	for _, pv := range predVariants {
		var runs []SingleRunResult
		for _, seed := range seeds {
			pred := pv.buildPred(seed)
			cfg := policy.DefaultVTCConfig()
			cfg.ChargingModel = policy.ChargingPredictedAtAdmission
			pol := policy.NewVTC(cfg)

			res := runSimulation(pol, pred, sim.GatewayConfig{Enabled: false}, 1.0, true, false, seed)
			runs = append(runs, res)
		}
		agg := aggregateRuns(pv.name, "predictor_heavy_tail", pv.param, runs)
		allResults = append(allResults, agg)
		fmt.Printf("%-32s | %-18s | %-18s | %-20s | %-16s | %-16s\n",
			agg.Name, agg.ChatWtdShare.StringPct(), agg.ChatRawShare.StringPct(),
			agg.ChatTTFTMeanMs.String(), agg.PredictorMAE.String(), agg.CounterDrift.String())
	}

	// -------------------------------------------------------------------------
	// 3. Engine In-Flight Cap K Sweep
	// -------------------------------------------------------------------------
	fmt.Println("\n==================================================================================================")
	fmt.Println("3. GATEWAY MODE: ENGINE IN-FLIGHT CAP K SWEEP")
	fmt.Println("Engine max batch size B = 16. Gateway limits concurrent engine in-flight requests to K.")
	fmt.Println("Mechanism: When K <= 16, gateway queues requests and schedules them just-in-time into the batch.")
	fmt.Println("           When K > 16, excess requests spill into engine internal FIFO queue, bypassing gateway fairness.")
	fmt.Println("--------------------------------------------------------------------------------------------------")
	fmt.Printf("%-20s | %-8s | %-22s | %-22s | %-18s | %-18s\n",
		"Configuration", "Cap K", "Victim TTFT Mean ms", "Victim TTFT P95 ms", "Chat Wtd Share", "Chat Raw Share")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	caps := []struct {
		k     int
		label string
	}{
		{4, "K=4"},
		{16, "K=16 (B)"},
		{64, "K=64"},
		{256, "K=256"},
		{10000, "unbounded"},
	}

	for _, c := range caps {
		var runs []SingleRunResult
		for _, seed := range seeds {
			gwCfg := sim.GatewayConfig{
				Enabled:      true,
				MaxInFlightK: c.k,
			}
			pol := policy.NewVTC(policy.DefaultVTCConfig())
			res := runSimulation(pol, nil, gwCfg, 1.0, false, false, seed)
			runs = append(runs, res)
		}
		agg := aggregateRuns(fmt.Sprintf("gateway-k-%s", c.label), "gateway_cap_k", c.label, runs)
		allResults = append(allResults, agg)
		fmt.Printf("%-20s | %-8s | %-22s | %-22s | %-18s | %-18s\n",
			agg.Name, c.label, agg.ChatTTFTMeanMs.String(), agg.ChatTTFTP95Ms.String(),
			agg.ChatWtdShare.StringPct(), agg.ChatRawShare.StringPct())
	}

	// -------------------------------------------------------------------------
	// 4. Capacity Load Sweep (~50%, 80%, 95%, 120%)
	// -------------------------------------------------------------------------
	fmt.Println("\n==================================================================================================")
	fmt.Println("4. CAPACITY LOAD SWEEP (~50%, 80%, 95%, 120% CAPACITY)")
	fmt.Println("Capacity Definition: Nominal load (1.0 = 100%) matches continuous batching saturation where arrival")
	fmt.Println("rate (Chat 50 req/s + RAG 10 req/s = 50,880 raw tok/s) equals maximum sustained engine throughput.")
	fmt.Println("Note: Policies allocate different token shares (e.g. FIFO gives Chat ~23% wtd share while VTC/DRR give")
	fmt.Println("      ~50%). TTFT comparisons reflect both queuing priority and allocated service share.")
	fmt.Println("--------------------------------------------------------------------------------------------------")
	fmt.Printf("%-10s | %-8s | %-20s | %-20s | %-20s | %-20s\n",
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

	loadPolicies := []struct {
		name      string
		buildPoly func() policy.Policy
	}{
		{"fifo", func() policy.Policy { return policy.NewFIFO() }},
		{"drr", func() policy.Policy { return policy.NewDRR(policy.DefaultDRRConfig()) }},
		{"vtc", func() policy.Policy { return policy.NewVTC(policy.DefaultVTCConfig()) }},
	}

	for _, lp := range loadPolicies {
		for _, l := range loads {
			var runs []SingleRunResult
			for _, seed := range seeds {
				pol := lp.buildPoly()
				res := runSimulation(pol, nil, sim.GatewayConfig{Enabled: false}, l.scale, false, false, seed)
				runs = append(runs, res)
			}
			agg := aggregateRuns(fmt.Sprintf("%s-%s", lp.name, l.label), "load_sweep", fmt.Sprintf("%s-%s", lp.name, l.label), runs)
			allResults = append(allResults, agg)
			fmt.Printf("%-10s | %-8s | %-20s | %-20s | %-20s | %-20s\n",
				lp.name, l.label, agg.ChatWtdShare.StringPct(), agg.ChatRawShare.StringPct(),
				agg.ChatTTFTMeanMs.String(), agg.RAGTTFTMeanMs.String())
		}
	}

	// -------------------------------------------------------------------------
	// 5. Priority-Band Experiment (Strict Inter-Tier Priority)
	// -------------------------------------------------------------------------
	fmt.Println("\n==================================================================================================")
	fmt.Println("5. PRIORITY-BAND EXPERIMENT: STRICT INTER-TIER PRIORITY SCHEDULING")
	fmt.Println("Setup: Tier 0 (High Priority / Interactive Chat) vs Tier 1 (Best Effort / Batch RAG).")
	fmt.Println("Mechanism: Tier 0 requests preempt Tier 1 from scheduling slots whenever backlogged.")
	fmt.Println("--------------------------------------------------------------------------------------------------")
	fmt.Printf("%-32s | %-20s | %-20s | %-16s | %-16s\n",
		"Configuration", "Tier 0 TTFT Mean ms", "Tier 1 TTFT Mean ms", "Tier 0 Wtd Share", "Tier 1 Wtd Share")
	fmt.Println("--------------------------------------------------------------------------------------------------")

	// 5a. Baseline: Flat VTC (No Priority, Full Load)
	{
		var runs []SingleRunResult
		for _, seed := range seeds {
			pol := policy.NewVTC(policy.DefaultVTCConfig())
			res := runSimulation(pol, nil, sim.GatewayConfig{Enabled: false}, 1.0, false, false, seed)
			runs = append(runs, res)
		}
		agg := aggregateRuns("baseline-no-priority", "priority_bands", "flat-vtc-full", runs)
		allResults = append(allResults, agg)
		fmt.Printf("%-32s | %-20s | %-20s | %-16s | %-16s\n",
			"Flat VTC (No Priority, 100% Load)", agg.ChatTTFTMeanMs.String(), agg.RAGTTFTMeanMs.String(),
			agg.ChatWtdShare.StringPct(), fmt.Sprintf("%.1f%%", (1.0-agg.ChatWtdShare.Mean)*100))
	}

	// 5b. Priority Bands: Full Overload (Tier 0 saturates capacity)
	{
		var runs []SingleRunResult
		for _, seed := range seeds {
			pbMap := map[int]policy.Policy{
				0: policy.NewVTC(policy.DefaultVTCConfig()),
				1: policy.NewVTC(policy.DefaultVTCConfig()),
			}
			pol := policy.NewPriorityBandPolicy(pbMap)
			res := runSimulation(pol, nil, sim.GatewayConfig{Enabled: false}, 1.0, false, true, seed)
			runs = append(runs, res)
		}
		agg := aggregateRuns("priority-bands-overload", "priority_bands", "t0-saturating-100", runs)
		allResults = append(allResults, agg)
		fmt.Printf("%-32s | %-20s | %-20s | %-16s | %-16s\n",
			"Priority Bands (T0 Overload)", agg.ChatTTFTMeanMs.String(), "Preempted / Stalled",
			agg.ChatWtdShare.StringPct(), fmt.Sprintf("%.1f%%", (1.0-agg.ChatWtdShare.Mean)*100))
	}

	// 5c. Priority Bands: Managed Tier 0 Load (Tier 0 takes ~40% load, Tier 1 gets residual capacity)
	{
		var runs []SingleRunResult
		for _, seed := range seeds {
			pbMap := map[int]policy.Policy{
				0: policy.NewVTC(policy.DefaultVTCConfig()),
				1: policy.NewVTC(policy.DefaultVTCConfig()),
			}
			pol := policy.NewPriorityBandPolicy(pbMap)
			res := runSimulation(pol, nil, sim.GatewayConfig{Enabled: false}, 0.60, false, true, seed)
			runs = append(runs, res)
		}
		agg := aggregateRuns("priority-bands-managed", "priority_bands", "t0-managed-60", runs)
		allResults = append(allResults, agg)
		fmt.Printf("%-32s | %-20s | %-20s | %-16s | %-16s\n",
			"Priority Bands (T0 Managed 60%)", agg.ChatTTFTMeanMs.String(), agg.RAGTTFTMeanMs.String(),
			agg.ChatWtdShare.StringPct(), fmt.Sprintf("%.1f%%", (1.0-agg.ChatWtdShare.Mean)*100))
	}


	// Save all results to experiments/results/exp_phase2a_summary.json
	os.MkdirAll("experiments/results", 0755)
	outPath := filepath.Join("experiments", "results", "exp_phase2a_summary.json")
	data, err := json.MarshalIndent(allResults, "", "  ")
	if err == nil {
		os.WriteFile(outPath, data, 0644)
		fmt.Printf("\nSaved complete Phase 2a evaluation matrix with CIs to: %s\n", outPath)
	}

	fmt.Println("\n==================================================================================================")
	fmt.Println("                      PHASE 2a BENCHMARK MATRIX COMPLETED SUCCESSFULLY                            ")
	fmt.Println("==================================================================================================")
}

// runSimulation executes a single seed simulation.
func runSimulation(
	pol policy.Policy,
	predictor policy.OutputPredictor,
	gwCfg sim.GatewayConfig,
	loadScale float64,
	heavyTailed bool,
	priorityMode bool,
	seed int64,
) SingleRunResult {
	serverCfg := sim.ServerConfig{
		MaxBatchSize:     16,
		KVCapacityTokens: 16384,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	server := sim.NewServer(serverCfg)
	qm := sim.NewTenantQueueManager()
	var engineQueue []*sim.Request

	chatRate := 50.0 * loadScale
	ragRate := 10.0 * loadScale
	horizonS := 30.0
	horizonUs := int64(horizonS * 1e6)
	warmupUs := int64(5.0 * 1e6)
	arrivalHorizonS := horizonS + 20.0

	rng := rand.New(rand.NewSource(seed))

	var trace []*sim.Request
	var predErrors []float64
	var counterDrifts []float64

	// Generate Chat arrivals
	var tUs float64 = 0
	reqID := 0
	for tUs < arrivalHorizonS*1e6 {
		intervalUs := (-math.Log(1.0-rng.Float64()) / chatRate) * 1e6
		tUs += intervalUs
		if tUs >= arrivalHorizonS*1e6 {
			break
		}
		reqID++
		id := fmt.Sprintf("chat-%05d", reqID)

		actualOut := 64
		if heavyTailed {
			// Log-normal: median 64, sigma 0.75, range [8, 512]
			val := math.Round(64.0 * math.Exp(0.75*rng.NormFloat64()))
			if val < 8 {
				val = 8
			} else if val > 512 {
				val = 512
			}
			actualOut = int(val)
		}

		predOut := actualOut
		if predictor != nil {
			predOut = predictor.Predict("chat", 128, actualOut)
			errVal := math.Abs(float64(predOut - actualOut))
			predErrors = append(predErrors, errVal)
			counterDrifts = append(counterDrifts, 2.0*errVal) // w_out = 2.0
		}

		req := sim.NewRequest(id, "chat", 128, actualOut, predOut, int64(tUs))
		if priorityMode {
			req.WithPriority(0) // Tier 0 = High
		}
		trace = append(trace, req)
	}

	// Generate RAG arrivals
	tUs = 0
	reqID = 0
	for tUs < arrivalHorizonS*1e6 {
		intervalUs := (-math.Log(1.0-rng.Float64()) / ragRate) * 1e6
		tUs += intervalUs
		if tUs >= arrivalHorizonS*1e6 {
			break
		}
		reqID++
		id := fmt.Sprintf("rag-%05d", reqID)

		actualOut := 128
		if heavyTailed {
			// Log-normal: median 128, sigma 0.75, range [16, 1024]
			val := math.Round(128.0 * math.Exp(0.75*rng.NormFloat64()))
			if val < 16 {
				val = 16
			} else if val > 1024 {
				val = 1024
			}
			actualOut = int(val)
		}

		predOut := actualOut
		if predictor != nil {
			predOut = predictor.Predict("rag", 4000, actualOut)
			errVal := math.Abs(float64(predOut - actualOut))
			predErrors = append(predErrors, errVal)
			counterDrifts = append(counterDrifts, 2.0*errVal)
		}

		req := sim.NewRequest(id, "rag", 4000, actualOut, predOut, int64(tUs))
		if priorityMode {
			req.WithPriority(1) // Tier 1 = Low
		}
		trace = append(trace, req)
	}

	// Sort trace stably by ArrivalTime, then ID
	sort.Slice(trace, func(i, j int) bool {
		if trace[i].ArrivalTime() == trace[j].ArrivalTime() {
			return trace[i].ID() < trace[j].ID()
		}
		return trace[i].ArrivalTime() < trace[j].ArrivalTime()
	})

	var nowUs int64 = 0
	traceIdx := 0
	rawChat := 0
	rawRAG := 0
	wtdChat := 0.0
	wtdRAG := 0.0

	var chatTTFTs, ragTTFTs []int64

	for nowUs < horizonUs {
		// 1. Enqueue arrivals up to nowUs
		for traceIdx < len(trace) && trace[traceIdx].ArrivalTime() <= nowUs {
			qm.Enqueue(trace[traceIdx], pol, serverCfg.KVCapacityTokens)
			traceIdx++
		}
		qm.SetCurrentTime(nowUs)

		// 2. Batch formation
		if gwCfg.Enabled {
			// Gateway mode
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
			// Direct admission (Break-on-no-fit)
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

		// 3. Execution step
		if server.RunningCount() > 0 {
			res := server.Step(pol, nowUs)
			nowUs += res.StepDurationUs

			for _, comp := range res.Completed {
				if comp.TenantID() == "chat" {
					chatTTFTs = append(chatTTFTs, comp.TTFT())
				} else {
					ragTTFTs = append(ragTTFTs, comp.TTFT())
				}
			}

			// Window accounting in [warmupUs, horizonUs]
			if nowUs >= warmupUs && nowUs <= horizonUs {
				rawChat += res.TokensPerTenant["chat"]
				rawRAG += res.TokensPerTenant["rag"]
				wtdChat += res.WeightedTokensPerTenant["chat"]
				wtdRAG += res.WeightedTokensPerTenant["rag"]
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

	meanChatTTFT := 0.0
	p95ChatTTFT := 0.0
	if len(chatTTFTs) > 0 {
		sort.Slice(chatTTFTs, func(i, j int) bool { return chatTTFTs[i] < chatTTFTs[j] })
		var sum int64
		for _, v := range chatTTFTs {
			sum += v
		}
		meanChatTTFT = float64(sum) / float64(len(chatTTFTs)) / 1000.0
		p95Idx := int(float64(len(chatTTFTs)) * 0.95)
		if p95Idx >= len(chatTTFTs) {
			p95Idx = len(chatTTFTs) - 1
		}
		p95ChatTTFT = float64(chatTTFTs[p95Idx]) / 1000.0
	}

	meanRAGTTFT := 0.0
	p95RAGTTFT := 0.0
	if len(ragTTFTs) > 0 {
		sort.Slice(ragTTFTs, func(i, j int) bool { return ragTTFTs[i] < ragTTFTs[j] })
		var sum int64
		for _, v := range ragTTFTs {
			sum += v
		}
		meanRAGTTFT = float64(sum) / float64(len(ragTTFTs)) / 1000.0
		p95Idx := int(float64(len(ragTTFTs)) * 0.95)
		if p95Idx >= len(ragTTFTs) {
			p95Idx = len(ragTTFTs) - 1
		}
		p95RAGTTFT = float64(ragTTFTs[p95Idx]) / 1000.0
	}

	meanMAE := 0.0
	if len(predErrors) > 0 {
		var sum float64
		for _, e := range predErrors {
			sum += e
		}
		meanMAE = sum / float64(len(predErrors))
	}

	meanDrift := 0.0
	if len(counterDrifts) > 0 {
		var sum float64
		for _, d := range counterDrifts {
			sum += d
		}
		meanDrift = sum / float64(len(counterDrifts))
	}

	return SingleRunResult{
		ChatRawShare:   chatRawShare,
		ChatWtdShare:   chatWtdShare,
		MaxMinRatio:    maxMin,
		ChatTTFTMeanMs: meanChatTTFT,
		ChatTTFTP95Ms:  p95ChatTTFT,
		RAGTTFTMeanMs:  meanRAGTTFT,
		RAGTTFTP95Ms:   p95RAGTTFT,
		PredictorMAE:   meanMAE,
		CounterDrift:   meanDrift,
	}
}
