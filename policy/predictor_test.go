package policy_test

import (
	"math"
	"testing"

	"fairlane/policy"
)

// TestPredictors verifies oracle, mean, and noisy oracle distributions.
func TestPredictors(t *testing.T) {
	// 1. Oracle predictor
	oracle := policy.NewOraclePredictor()
	if pred := oracle.Predict("t1", 100, 256); pred != 256 {
		t.Fatalf("Oracle expected 256, got %d", pred)
	}

	// 2. Mean predictor
	meanPred := policy.NewMeanPredictor(64)
	if pred := meanPred.Predict("t1", 100, 256); pred != 64 {
		t.Fatalf("Mean initial prior expected 64, got %d", pred)
	}
	meanPred.RecordCompletion("t1", 100)
	meanPred.RecordCompletion("t1", 200)
	if pred := meanPred.Predict("t1", 100, 256); pred != 150 {
		t.Fatalf("Mean after completions expected 150, got %d", pred)
	}

	// 3. Noisy oracle predictor
	// L_true = 100, bias = 0.20 (+20%), sigma = 0.1
	// Over 10,000 samples, sample mean should be close to 100 * 1.20 = 120.
	noisy := policy.NewNoisyOraclePredictor(0.1, 0.20, 42)
	sum := 0.0
	n := 10000
	for i := 0; i < n; i++ {
		pred := noisy.Predict("t1", 50, 100)
		sum += float64(pred)
	}
	empMean := sum / float64(n)
	expectedMean := 120.0
	if math.Abs(empMean-expectedMean) > 3.0 {
		t.Fatalf("Noisy oracle empirical mean %f diverged from expected %f", empMean, expectedMean)
	}
}

// TestVTCChargingVariants tests VTC charging under incremental, completion, and predicted-at-admission.
func TestVTCChargingVariants(t *testing.T) {
	// 1. Incremental (VTC native)
	cfgInc := policy.DefaultVTCConfig()
	cfgInc.ChargingModel = policy.ChargingIncremental
	vtcInc := policy.NewVTC(cfgInc)

	req1 := &mockRequest{id: "r1", tenantID: "t1", input: 100, predOut: 50}
	vtcInc.OnArrival(req1, true, "", nil)
	vtcInc.OnDispatch(req1)
	// Input weight = 1.0 -> counter = 100
	if c := vtcInc.Counter("t1"); c != 100.0 {
		t.Fatalf("Expected counter 100 after incremental prefill, got %f", c)
	}
	// Decode 20 tokens -> Output weight = 2.0 -> +40
	vtcInc.OnTokens("t1", 0, 20)
	if c := vtcInc.Counter("t1"); c != 140.0 {
		t.Fatalf("Expected counter 140 after decode tokens, got %f", c)
	}
	vtcInc.OnComplete("t1", req1, 100, 20)
	if c := vtcInc.Counter("t1"); c != 140.0 {
		t.Fatalf("Expected counter 140 after completion in incremental mode, got %f", c)
	}

	// 2. Predicted at admission + correct on completion
	cfgPred := policy.DefaultVTCConfig()
	cfgPred.ChargingModel = policy.ChargingPredictedAtAdmission
	vtcPred := policy.NewVTC(cfgPred)

	req2 := &mockRequest{id: "r2", tenantID: "t1", input: 100, predOut: 50}
	vtcPred.OnArrival(req2, true, "", nil)
	vtcPred.OnDispatch(req2)
	// Charged at admission: 100*1 + 50*2 = 200
	if c := vtcPred.Counter("t1"); c != 200.0 {
		t.Fatalf("Expected counter 200 at admission with predicted output, got %f", c)
	}
	// Actual completion had 70 output tokens (20 more than predicted)
	// Correction = 20 * 2.0 = +40 -> final counter = 240
	vtcPred.OnComplete("t1", req2, 100, 70)
	if c := vtcPred.Counter("t1"); c != 240.0 {
		t.Fatalf("Expected counter 240 after correction on completion, got %f", c)
	}
}
