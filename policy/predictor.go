package policy

import (
	"math"
	"math/rand"
	"sync"
)

// OutputPredictor generates predicted output token counts for requests before scheduling.
// CRITICAL: Policy implementations only observe PredictedOutputTokens(), never true output tokens.
type OutputPredictor interface {
	Name() string
	Predict(tenantID string, inputTokens int, trueOutputTokens int) int
}

// OraclePredictor perfectly predicts the true output tokens.
type OraclePredictor struct{}

func NewOraclePredictor() *OraclePredictor {
	return &OraclePredictor{}
}

func (p *OraclePredictor) Name() string {
	return "oracle"
}

func (p *OraclePredictor) Predict(tenantID string, inputTokens int, trueOutputTokens int) int {
	if trueOutputTokens < 1 {
		return 1
	}
	return trueOutputTokens
}

// MeanPredictor predicts the historical per-tenant mean output length.
type MeanPredictor struct {
	mu           sync.Mutex
	defaultMean  int
	tenantTotals map[string]int
	tenantCounts map[string]int
}

func NewMeanPredictor(defaultMean int) *MeanPredictor {
	if defaultMean < 1 {
		defaultMean = 64
	}
	return &MeanPredictor{
		defaultMean:  defaultMean,
		tenantTotals: make(map[string]int),
		tenantCounts: make(map[string]int),
	}
}

func (p *MeanPredictor) Name() string {
	return "mean-per-tenant"
}

func (p *MeanPredictor) RecordCompletion(tenantID string, outputTokens int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tenantTotals[tenantID] += outputTokens
	p.tenantCounts[tenantID]++
}

func (p *MeanPredictor) SetTenantPrior(tenantID string, mean int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tenantTotals[tenantID] = mean
	p.tenantCounts[tenantID] = 1
}

func (p *MeanPredictor) Predict(tenantID string, inputTokens int, trueOutputTokens int) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	count := p.tenantCounts[tenantID]
	if count > 0 {
		return p.tenantTotals[tenantID] / count
	}
	return p.defaultMean
}

// NoisyOraclePredictor models imperfect estimation with multiplicative lognormal noise and systematic bias.
// L_hat = max(1, round(L_true * (1 + bias) * exp(sigma * Z - sigma^2 / 2)))
// where Z ~ N(0, 1), E[exp(sigma*Z - sigma^2/2)] = 1.
type NoisyOraclePredictor struct {
	mu    sync.Mutex
	sigma float64
	bias  float64
	rng   *rand.Rand
}

func NewNoisyOraclePredictor(sigma, bias float64, seed int64) *NoisyOraclePredictor {
	return &NoisyOraclePredictor{
		sigma: sigma,
		bias:  bias,
		rng:   rand.New(rand.NewSource(seed)),
	}
}

func (p *NoisyOraclePredictor) Name() string {
	return "noisy-oracle"
}

func (p *NoisyOraclePredictor) Predict(tenantID string, inputTokens int, trueOutputTokens int) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	if trueOutputTokens < 1 {
		trueOutputTokens = 1
	}

	z := p.rng.NormFloat64()
	logFactor := math.Exp(p.sigma*z - (p.sigma*p.sigma)/2.0)
	multiplier := (1.0 + p.bias) * logFactor
	predicted := int(math.Round(float64(trueOutputTokens) * multiplier))
	if predicted < 1 {
		predicted = 1
	}
	return predicted
}
