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

type SummaryResult struct {
	Name            string             `json:"name"`
	Category        string             `json:"category"`
	Parameter       string             `json:"parameter"`
	ChatRawShare    float64            `json:"chat_raw_share"`
	ChatWtdShare    float64            `json:"chat_wtd_share"`
	MaxMinRatio     float64            `json:"max_min_ratio"`
	ChatTTFTMeanMs  float64            `json:"chat_ttft_mean_ms"`
	RAGTTFTMeanMs   float64            `json:"rag_ttft_mean_ms"`
	TotalTokens     int                `json:"total_tokens"`
	PerTenantTokens map[string]int     `json:"per_tenant_tokens"`
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("             FAIRLANE PHASE 2a BENCHMARK MATRIX EVALUATION                     ")
	fmt.Println("================================================================================")

	var allResults []SummaryResult

	// -------------------------------------------------------------------------
	// 1. DRR Token Quantum & Weight Sweep
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 1. DEFICIT ROUND ROBIN (DRR) TOKEN QUANTUM & WEIGHT EVALUATION ---")
	fmt.Printf("%-24s | %-10s | %-12s | %-12s | %-10s\n", "Configuration", "Quantum", "Chat Raw (%)", "Chat Wtd (%)", "Max-Min")
	fmt.Println("--------------------------------------------------------------------------------")

	quantums := []int{256, 512, 1024, 2048}
	for _, q := range quantums {
		cfg := policy.DefaultDRRConfig()
		cfg.BaseQuantumTokens = q
		pol := policy.NewDRR(cfg)

		res := runHeterogeneousSimulation(pol, nil, sim.GatewayConfig{Enabled: false}, 1.0)
		res.Category = "drr_quantum"
		res.Parameter = fmt.Sprintf("Q=%d", q)
		res.Name = fmt.Sprintf("drr-q%d", q)
		allResults = append(allResults, res)

		fmt.Printf("%-24s | %-10d | %11.1f%% | %11.1f%% | %9.2f\n",
			res.Name, q, res.ChatRawShare*100, res.ChatWtdShare*100, res.MaxMinRatio)
	}

	// DRR with Weights (Chat weight = 2.0, RAG weight = 1.0)
	cfgWeighted := policy.DefaultDRRConfig()
	cfgWeighted.BaseQuantumTokens = 512
	cfgWeighted.TenantWeights = map[string]float64{"chat": 2.0, "rag": 1.0}
	drrWeighted := policy.NewDRR(cfgWeighted)
	resWeighted := runHeterogeneousSimulation(drrWeighted, nil, sim.GatewayConfig{Enabled: false}, 1.0)
	resWeighted.Category = "drr_weighted"
	resWeighted.Parameter = "w_chat=2.0,w_rag=1.0"
	resWeighted.Name = "drr-weighted-2x"
	allResults = append(allResults, resWeighted)
	fmt.Printf("%-24s | %-10s | %11.1f%% | %11.1f%% | %9.2f\n",
		resWeighted.Name, "512 (w=2:1)", resWeighted.ChatRawShare*100, resWeighted.ChatWtdShare*100, resWeighted.MaxMinRatio)

	// -------------------------------------------------------------------------
	// 2. Predictors & Charging Variants Sweep
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. PREDICTORS & CHARGING VARIANTS (VTC) ---")
	fmt.Printf("%-28s | %-16s | %-12s | %-12s | %-10s\n", "Policy Variant", "Charging Model", "Chat Raw (%)", "Chat Wtd (%)", "Max-Min")
	fmt.Println("--------------------------------------------------------------------------------")

	// 2a. Incremental charging (VTC native)
	cfgInc := policy.DefaultVTCConfig()
	cfgInc.ChargingModel = policy.ChargingIncremental
	resInc := runHeterogeneousSimulation(policy.NewVTC(cfgInc), nil, sim.GatewayConfig{Enabled: false}, 1.0)
	resInc.Category = "charging_variants"
	resInc.Parameter = "incremental"
	resInc.Name = "vtc-incremental-native"
	allResults = append(allResults, resInc)
	fmt.Printf("%-28s | %-16s | %11.1f%% | %11.1f%% | %9.2f\n",
		resInc.Name, "incremental", resInc.ChatRawShare*100, resInc.ChatWtdShare*100, resInc.MaxMinRatio)

	// 2b. Completion charging
	cfgComp := policy.DefaultVTCConfig()
	cfgComp.ChargingModel = policy.ChargingCompletion
	resComp := runHeterogeneousSimulation(policy.NewVTC(cfgComp), nil, sim.GatewayConfig{Enabled: false}, 1.0)
	resComp.Category = "charging_variants"
	resComp.Parameter = "completion"
	resComp.Name = "vtc-completion-charging"
	allResults = append(allResults, resComp)
	fmt.Printf("%-28s | %-16s | %11.1f%% | %11.1f%% | %9.2f\n",
		resComp.Name, "completion", resComp.ChatRawShare*100, resComp.ChatWtdShare*100, resComp.MaxMinRatio)

	// 2c. Predicted at admission + correct on completion with different predictors
	predictors := []struct {
		name string
		pred policy.OutputPredictor
	}{
		{"oracle", policy.NewOraclePredictor()},
		{"mean-prior", policy.NewMeanPredictor(64)},
		{"noisy-oracle (+20% bias)", policy.NewNoisyOraclePredictor(0.3, 0.20, 42)},
		{"noisy-oracle (-20% bias)", policy.NewNoisyOraclePredictor(0.3, -0.20, 42)},
	}

	for _, p := range predictors {
		cfgPred := policy.DefaultVTCConfig()
		cfgPred.ChargingModel = policy.ChargingPredictedAtAdmission
		pol := policy.NewVTC(cfgPred)

		res := runHeterogeneousSimulation(pol, p.pred, sim.GatewayConfig{Enabled: false}, 1.0)
		res.Category = "predicted_admission"
		res.Parameter = p.name
		res.Name = fmt.Sprintf("vtc-pred-%s", p.name)
		allResults = append(allResults, res)
		fmt.Printf("%-28s | %-16s | %11.1f%% | %11.1f%% | %9.2f\n",
			res.Name, "pred-admission", res.ChatRawShare*100, res.ChatWtdShare*100, res.MaxMinRatio)
	}

	// -------------------------------------------------------------------------
	// 3. Gateway In-Flight Cap K Sweep
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. GATEWAY MODE: ENGINE IN-FLIGHT CAP K SWEEP ---")
	fmt.Printf("%-24s | %-8s | %-12s | %-12s | %-10s\n", "Configuration", "Cap K", "Chat Raw (%)", "Chat Wtd (%)", "Max-Min")
	fmt.Println("--------------------------------------------------------------------------------")

	caps := []int{4, 8, 16, 32}
	for _, capK := range caps {
		gwCfg := sim.GatewayConfig{
			Enabled:      true,
			MaxInFlightK: capK,
		}
		pol := policy.NewVTC(policy.DefaultVTCConfig())
		res := runHeterogeneousSimulation(pol, nil, gwCfg, 1.0)
		res.Category = "gateway_cap_k"
		res.Parameter = fmt.Sprintf("K=%d", capK)
		res.Name = fmt.Sprintf("gateway-vtc-k%d", capK)
		allResults = append(allResults, res)

		fmt.Printf("%-24s | %-8d | %11.1f%% | %11.1f%% | %9.2f\n",
			res.Name, capK, res.ChatRawShare*100, res.ChatWtdShare*100, res.MaxMinRatio)
	}

	// -------------------------------------------------------------------------
	// 4. Capacity Load Sweep (~50%, 80%, 95%, 120%)
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. CAPACITY LOAD SWEEP (~50%, 80%, 95%, 120% CAPACITY) ---")
	fmt.Printf("%-18s | %-8s | %-12s | %-12s | %-14s | %-14s\n",
		"Policy", "Load (%)", "Chat Raw (%)", "Chat Wtd (%)", "Chat TTFT (ms)", "RAG TTFT (ms)")
	fmt.Println("------------------------------------------------------------------------------------------------")

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
		name string
		pol  policy.Policy
	}{
		{"fifo", policy.NewFIFO()},
		{"drr", policy.NewDRR(policy.DefaultDRRConfig())},
		{"vtc", policy.NewVTC(policy.DefaultVTCConfig())},
	}

	for _, lp := range loadPolicies {
		for _, l := range loads {
			res := runHeterogeneousSimulation(lp.pol, nil, sim.GatewayConfig{Enabled: false}, l.scale)
			res.Category = "load_sweep"
			res.Parameter = fmt.Sprintf("%s-load-%s", lp.name, l.label)
			res.Name = fmt.Sprintf("%s-%s", lp.name, l.label)
			allResults = append(allResults, res)

			fmt.Printf("%-18s | %-8s | %11.1f%% | %11.1f%% | %14.1f | %14.1f\n",
				lp.name, l.label, res.ChatRawShare*100, res.ChatWtdShare*100, res.ChatTTFTMeanMs, res.RAGTTFTMeanMs)
		}
	}

	// Save all results to experiments/results/exp_phase2a_summary.json
	os.MkdirAll("experiments/results", 0755)
	outPath := filepath.Join("experiments", "results", "exp_phase2a_summary.json")
	data, err := json.MarshalIndent(allResults, "", "  ")
	if err == nil {
		os.WriteFile(outPath, data, 0644)
		fmt.Printf("\nSaved complete Phase 2a evaluation matrix to: %s\n", outPath)
	}

	fmt.Println("\n================================================================================")
	fmt.Println("             PHASE 2a BENCHMARK MATRIX COMPLETED SUCCESSFULLY                   ")
	fmt.Println("================================================================================")
}

// runHeterogeneousSimulation executes a synthetic workload trace with Chat vs RAG tenants.
func runHeterogeneousSimulation(
	pol policy.Policy,
	predictor policy.OutputPredictor,
	gwCfg sim.GatewayConfig,
	loadScale float64,
) SummaryResult {
	serverCfg := sim.ServerConfig{
		MaxBatchSize:     16,
		KVCapacityTokens: 16384,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	server := sim.NewServer(serverCfg)
	qm := sim.NewTenantQueueManager()
	var engineQueue []*sim.Request

	// Workload parameters:
	// Chat: prompt 128, decode 64. Nominal 50 req/s.
	// RAG:  prompt 4000, decode 128. Nominal 10 req/s.
	chatRate := 50.0 * loadScale
	ragRate := 10.0 * loadScale
	horizonS := 30.0
	horizonUs := int64(horizonS * 1e6)
	warmupUs := int64(5.0 * 1e6)

	// Generate extra arrivals past the horizon to ensure backlog
	arrivalHorizonS := horizonS + 20.0

	rng := rand.New(rand.NewSource(12345))

	var trace []*sim.Request

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
		predOut := 64
		if predictor != nil {
			predOut = predictor.Predict("chat", 128, actualOut)
		}
		trace = append(trace, sim.NewRequest(id, "chat", 128, actualOut, predOut, int64(tUs)))
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
		predOut := 128
		if predictor != nil {
			predOut = predictor.Predict("rag", 4000, actualOut)
		}
		trace = append(trace, sim.NewRequest(id, "rag", 4000, actualOut, predOut, int64(tUs)))
	}

	// Sort trace by arrival timestamp
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
			for len(engineQueue)+server.RunningCount() < gwCfg.MaxInFlightK && qm.TotalQueuedRequests() > 0 {
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
	if len(chatTTFTs) > 0 {
		var sum int64
		for _, v := range chatTTFTs {
			sum += v
		}
		meanChatTTFT = float64(sum) / float64(len(chatTTFTs)) / 1000.0 // ms
	}

	meanRAGTTFT := 0.0
	if len(ragTTFTs) > 0 {
		var sum int64
		for _, v := range ragTTFTs {
			sum += v
		}
		meanRAGTTFT = float64(sum) / float64(len(ragTTFTs)) / 1000.0 // ms
	}

	return SummaryResult{
		ChatRawShare:   chatRawShare,
		ChatWtdShare:   chatWtdShare,
		MaxMinRatio:    maxMin,
		ChatTTFTMeanMs: meanChatTTFT,
		RAGTTFTMeanMs:  meanRAGTTFT,
		TotalTokens:    rawChat + rawRAG,
		PerTenantTokens: map[string]int{
			"chat": rawChat,
			"rag":  rawRAG,
		},
	}
}
