package sim

import (
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
)

// TenantWorkloadConfig defines the traffic profile of an individual tenant.
type TenantWorkloadConfig struct {
	TenantID        string
	RequestCount    int
	ArrivalProcess  string  // "poisson", "bursty", "periodic"
	MeanIntervalUs  float64 // Mean inter-arrival time in microseconds
	BurstSize       int     // For bursty arrivals: requests per burst
	PromptLengthMin int
	PromptLengthMax int
	OutputLengthMin int
	OutputLengthMax int
}

// WorkloadGenerator produces deterministic request traces.
type WorkloadGenerator struct {
	masterSeed int64
}

func NewWorkloadGenerator(masterSeed int64) *WorkloadGenerator {
	return &WorkloadGenerator{
		masterSeed: masterSeed,
	}
}

// tenantSeed derives a unique, isolated seed per tenant stream.
// INVARIANT: Changing one tenant configuration does not perturb another tenant's RNG sequence.
func (g *WorkloadGenerator) tenantSeed(tenantID string) int64 {
	h := fnv.New64a()
	h.Write([]byte(tenantID))
	return g.masterSeed ^ int64(h.Sum64())
}

// GenerateTrace creates the complete set of arriving requests across all configured tenants.
func (g *WorkloadGenerator) GenerateTrace(configs []TenantWorkloadConfig) []*Request {
	var allRequests []*Request

	for _, cfg := range configs {
		rng := rand.New(rand.NewSource(g.tenantSeed(cfg.TenantID)))

		var curTimeUs int64 = 0

		for i := 0; i < cfg.RequestCount; i++ {
			// Compute inter-arrival time
			var interArrivalUs int64

			switch cfg.ArrivalProcess {
			case "poisson":
				// Standard exponential inter-arrival: -mean * ln(U) where U ~ Uniform(0, 1)
				u := rng.Float64()
				for u <= 0 || u >= 1.0 {
					u = rng.Float64()
				}
				interArrivalUs = int64(-cfg.MeanIntervalUs * math.Log(u))
				if interArrivalUs < 1 {
					interArrivalUs = 1
				}
			case "bursty":
				// Bursty: small intervals within a burst, large gap between bursts
				if cfg.BurstSize > 1 && (i%cfg.BurstSize != 0) {
					interArrivalUs = int64(rng.Intn(500) + 100) // 100-600 µs within burst
				} else {
					interArrivalUs = int64(cfg.MeanIntervalUs * float64(cfg.BurstSize))
				}
			case "periodic":
				interArrivalUs = int64(cfg.MeanIntervalUs)
			default:
				interArrivalUs = int64(cfg.MeanIntervalUs)
			}

			curTimeUs += interArrivalUs

			// Generate prompt and output length
			inputLen := cfg.PromptLengthMin
			if cfg.PromptLengthMax > cfg.PromptLengthMin {
				inputLen += rng.Intn(cfg.PromptLengthMax - cfg.PromptLengthMin + 1)
			}

			outputLen := cfg.OutputLengthMin
			if cfg.OutputLengthMax > cfg.OutputLengthMin {
				outputLen += rng.Intn(cfg.OutputLengthMax - cfg.OutputLengthMin + 1)
			}

			reqID := fmt.Sprintf("%s-req-%05d", cfg.TenantID, i+1)
			req := NewRequest(reqID, cfg.TenantID, inputLen, outputLen, 0, curTimeUs)
			allRequests = append(allRequests, req)
		}
	}

	return allRequests
}
