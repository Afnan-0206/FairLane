package main

import (
	"fmt"
	"math/rand"
	"sort"

	"fairlane/policy"
	"fairlane/sim"
)

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

func main() {
	fmt.Println("=== PHASE 1b DIAGNOSTIC INVESTIGATION ===")
	runDiagnosis()
}

func runDiagnosis() {
	const (
		numSeeds        = 30
		seedStart int64 = 100
	)

	serverCfg := sim.ServerConfig{
		KVCapacityTokens: 4096,
		MaxBatchSize:     16,
		BaseStepTimeUs:   5000,
		PerTokenCostUs:   5,
	}

	type Result struct {
		name           string
		victimSlowdown []float64
		victimTTFTMs   []float64
	}

	results := map[string]*Result{
		"rr":           {name: "Round-Robin"},
		"vtc-default":  {name: "VTC Default (flooder < victim)"},
		"vtc-swapped":  {name: "VTC Swapped Names (a_victim < z_flooder)"},
		"vtc-rand-tie": {name: "VTC Seeded Random Tie-Break"},
	}

	for s := int64(0); s < numSeeds; s++ {
		seed := seedStart + s
		gen := sim.NewWorkloadGenerator(seed)

		soloCfg := sim.TenantWorkloadConfig{
			TenantID:        "victim",
			RequestCount:    60,
			ArrivalProcess:  "poisson",
			MeanIntervalUs:  50000,
			PromptLengthMin: 128,
			PromptLengthMax: 128,
			OutputLengthMin: 64,
			OutputLengthMax: 64,
		}
		soloTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{soloCfg})
		soloSim := sim.NewSimulator(serverCfg, policy.NewFIFO(), 1000)
		soloMet := soloSim.Run(soloTrace)
		victimSoloLat := soloMet.PerTenant["victim"].TotalLatencyUs.Mean

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
		colocTrace := gen.GenerateTrace([]sim.TenantWorkloadConfig{soloCfg, flooderCfg})

		swappedVictimCfg := soloCfg
		swappedVictimCfg.TenantID = "a_victim"
		swappedFlooderCfg := flooderCfg
		swappedFlooderCfg.TenantID = "z_flooder"
		genSwapped := sim.NewWorkloadGenerator(seed)
		swappedTrace := genSwapped.GenerateTrace([]sim.TenantWorkloadConfig{swappedVictimCfg, swappedFlooderCfg})

		// 1. Round-Robin
		{
			simRR := sim.NewSimulator(serverCfg, policy.NewRoundRobin(), 1000)
			metRR := simRR.Run(colocTrace)
			vMet := metRR.PerTenant["victim"]
			slowdown := vMet.TotalLatencyUs.Mean / victimSoloLat
			results["rr"].victimSlowdown = append(results["rr"].victimSlowdown, slowdown)
			results["rr"].victimTTFTMs = append(results["rr"].victimTTFTMs, vMet.TTFTUs.Mean/1000.0)
		}

		// 2. VTC Default
		{
			simVTC := sim.NewSimulator(serverCfg, policy.NewVTC(policy.DefaultVTCConfig()), 1000)
			metVTC := simVTC.Run(colocTrace)
			vMet := metVTC.PerTenant["victim"]
			slowdown := vMet.TotalLatencyUs.Mean / victimSoloLat
			results["vtc-default"].victimSlowdown = append(results["vtc-default"].victimSlowdown, slowdown)
			results["vtc-default"].victimTTFTMs = append(results["vtc-default"].victimTTFTMs, vMet.TTFTUs.Mean/1000.0)
		}

		// 3. VTC Swapped Names
		{
			simVTC := sim.NewSimulator(serverCfg, policy.NewVTC(policy.DefaultVTCConfig()), 1000)
			metVTC := simVTC.Run(swappedTrace)
			vMet := metVTC.PerTenant["a_victim"]
			slowdown := vMet.TotalLatencyUs.Mean / victimSoloLat
			results["vtc-swapped"].victimSlowdown = append(results["vtc-swapped"].victimSlowdown, slowdown)
			results["vtc-swapped"].victimTTFTMs = append(results["vtc-swapped"].victimTTFTMs, vMet.TTFTUs.Mean/1000.0)
		}

		// 4. VTC Random Tie
		{
			polRand := NewRandomTieVTC(policy.DefaultVTCConfig(), seed)
			simVTC := sim.NewSimulator(serverCfg, polRand, 1000)
			metVTC := simVTC.Run(colocTrace)
			vMet := metVTC.PerTenant["victim"]
			slowdown := vMet.TotalLatencyUs.Mean / victimSoloLat
			results["vtc-rand-tie"].victimSlowdown = append(results["vtc-rand-tie"].victimSlowdown, slowdown)
			results["vtc-rand-tie"].victimTTFTMs = append(results["vtc-rand-tie"].victimTTFTMs, vMet.TTFTUs.Mean/1000.0)
		}
	}

	mean := func(slice []float64) float64 {
		sum := 0.0
		for _, v := range slice {
			sum += v
		}
		return sum / float64(len(slice))
	}

	fmt.Printf("\n| %-35s | %-16s | %-18s |\n", "Configuration", "Victim Slowdown", "Victim TTFT (ms)")
	fmt.Printf("|-------------------------------------|------------------|--------------------|\n")
	order := []string{"rr", "vtc-default", "vtc-swapped", "vtc-rand-tie"}
	for _, k := range order {
		r := results[k]
		fmt.Printf("| %-35s | %5.2fx            | %6.1f ms          |\n",
			r.name, mean(r.victimSlowdown), mean(r.victimTTFTMs))
	}

	diagnoseTraceDetails(serverCfg)
}

func diagnoseTraceDetails(serverCfg sim.ServerConfig) {
	fmt.Println("\n--- DETAILED TTFT DECOMPOSITION & BREAK-ON-NO-FIT (Seed 100) ---")
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

	for _, polName := range []string{"round-robin", "vtc"} {
		var pol policy.Policy
		if polName == "round-robin" {
			pol = policy.NewRoundRobin()
		} else {
			pol = policy.NewVTC(policy.DefaultVTCConfig())
		}

		server := sim.NewServer(serverCfg)
		qm := sim.NewTenantQueueManager()

		var nowUs int64 = 0
		traceIdx := 0

		type ReqDiag struct {
			arrUs            int64
			dispUs           int64
			firstTokUs       int64
			admitsWhileQueue int
			breakBlocks      int
		}
		diagMap := make(map[string]*ReqDiag)

		for traceIdx < len(trace) || qm.TotalQueuedRequests() > 0 || server.RunningCount() > 0 {
			for traceIdx < len(trace) && trace[traceIdx].ArrivalTime() <= nowUs {
				req := trace[traceIdx]
				qm.Enqueue(req, pol, serverCfg.KVCapacityTokens)
				if req.TenantID() == "victim" {
					diagMap[req.ID()] = &ReqDiag{arrUs: req.ArrivalTime()}
				}
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

				victimWaiting := qm.QueueDepth("victim") > 0

				if server.CanAdmit(head) {
					req := qm.PopHead(winner)
					server.Admit(req, pol, nowUs)
					if req.TenantID() == "victim" {
						diagMap[req.ID()].dispUs = nowUs
					} else if victimWaiting {
						for _, d := range diagMap {
							if d.dispUs == 0 {
								d.admitsWhileQueue++
							}
						}
					}
				} else {
					if victimWaiting && winner == "flooder" {
						for _, d := range diagMap {
							if d.dispUs == 0 {
								d.breakBlocks++
							}
						}
					}
					break
				}
			}

			if server.RunningCount() > 0 {
				res := server.Step(pol, nowUs)
				nowUs += res.StepDurationUs
				for _, comp := range res.Completed {
					if comp.TenantID() == "victim" {
						diagMap[comp.ID()].firstTokUs = comp.FirstTokenTime()
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

		var totalTTFT, totalQueueWait, totalPrefill int64
		var totalAdmitsWaited, totalBreakBlocks int
		n := len(diagMap)

		for _, d := range diagMap {
			queueWait := d.dispUs - d.arrUs
			prefill := d.firstTokUs - d.dispUs
			ttft := d.firstTokUs - d.arrUs

			totalTTFT += ttft
			totalQueueWait += queueWait
			totalPrefill += prefill
			totalAdmitsWaited += d.admitsWhileQueue
			totalBreakBlocks += d.breakBlocks
		}

		fmt.Printf("\n[%s]\n", polName)
		fmt.Printf("  Mean TTFT:               %8.2f ms (100.0%%)\n", float64(totalTTFT)/float64(n)/1000.0)
		fmt.Printf("    -> Queue Wait:         %8.2f ms (%5.1f%%)\n", float64(totalQueueWait)/float64(n)/1000.0, float64(totalQueueWait)/float64(totalTTFT)*100.0)
		fmt.Printf("    -> Prefill Duration:   %8.2f ms (%5.1f%%)\n", float64(totalPrefill)/float64(n)/1000.0, float64(totalPrefill)/float64(totalTTFT)*100.0)
		fmt.Printf("  Mean Flooder Admits Waited:    %6.1f per victim req\n", float64(totalAdmitsWaited)/float64(n))
		fmt.Printf("  Mean Break-On-No-Fit Blocks:   %6.1f per victim req\n", float64(totalBreakBlocks)/float64(n))
	}
}
