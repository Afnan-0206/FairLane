package sim

import (
	"math"
	"sort"
)

// Percentiles calculates p50, p95, p99 from a slice of int64 values.
type Percentiles struct {
	Count int
	Min   float64
	Mean  float64
	P50   float64
	P95   float64
	P99   float64
	Max   float64
}

func CalculatePercentiles(values []int64) Percentiles {
	n := len(values)
	if n == 0 {
		return Percentiles{}
	}

	sorted := make([]int64, n)
	copy(sorted, values)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var sum int64
	for _, v := range sorted {
		sum += v
	}

	getPercentile := func(p float64) float64 {
		if n == 1 {
			return float64(sorted[0])
		}
		idx := p * float64(n-1)
		low := int(math.Floor(idx))
		high := int(math.Ceil(idx))
		if low == high {
			return float64(sorted[low])
		}
		weight := idx - float64(low)
		return float64(sorted[low])*(1-weight) + float64(sorted[high])*weight
	}

	return Percentiles{
		Count: n,
		Min:   float64(sorted[0]),
		Mean:  float64(sum) / float64(n),
		P50:   getPercentile(0.50),
		P95:   getPercentile(0.95),
		P99:   getPercentile(0.99),
		Max:   float64(sorted[n-1]),
	}
}

// JainsIndex computes Jain's Fairness Index for a slice of allocations:
// J(x) = (\sum x_i)^2 / (n * \sum x_i^2)
// Values lie in [1/n, 1.0], where 1.0 represents perfect fairness.
func JainsIndex(allocations []float64) float64 {
	n := float64(len(allocations))
	if n <= 1 {
		return 1.0
	}

	var sum, sumSq float64
	for _, x := range allocations {
		sum += x
		sumSq += x * x
	}

	if sumSq == 0 {
		return 1.0
	}

	return (sum * sum) / (n * sumSq)
}

// MaxMinRatio computes the ratio between the maximum and minimum service allocations:
// max(x) / min(x). If min(x) == 0, returns +Inf.
func MaxMinRatio(allocations []float64) float64 {
	if len(allocations) == 0 {
		return 1.0
	}

	maxVal := -1.0
	minVal := math.MaxFloat64

	for _, x := range allocations {
		if x > maxVal {
			maxVal = x
		}
		if x < minVal {
			minVal = x
		}
	}

	if minVal <= 0 {
		if maxVal == 0 {
			return 1.0
		}
		return math.Inf(1)
	}

	return maxVal / minVal
}

// JainsDemandCapped computes Jain's index over demand-capped shares.
// For total service C and N active tenants with nominal fair share C/N,
// each tenant's target is min(demand_i, C/N).
// Normalized share is x_i = served_i / target_i.
func JainsDemandCapped(served []float64, demands []float64) float64 {
	n := len(served)
	if n <= 1 {
		return 1.0
	}
	totalServed := 0.0
	for _, s := range served {
		totalServed += s
	}
	if totalServed == 0 {
		return 1.0
	}
	fairShare := totalServed / float64(n)

	normalized := make([]float64, n)
	for i := 0; i < n; i++ {
		target := fairShare
		if i < len(demands) && demands[i] > 0 && demands[i] < fairShare {
			target = demands[i]
		}
		if target > 0 {
			normalized[i] = served[i] / target
		} else {
			normalized[i] = 1.0
		}
	}
	return JainsIndex(normalized)
}

// TenantMetrics summarizes performance and fairness for a single tenant.
type TenantMetrics struct {
	TenantID          string
	RequestsSubmitted int
	RequestsCompleted int
	NeverFitsRejected int
	InputTokensServed int
	OutputTokensServed int
	TotalTokensServed int
	TTFTUs            Percentiles
	ITLUs             Percentiles
	TotalLatencyUs    Percentiles
}

// SimulationMetrics summarizes overall simulation execution.
type SimulationMetrics struct {
	PolicyName             string
	TotalVirtualTimeUs     int64
	IdleWhileQueuedUs      int64 // Invariant: must be 0 for work-conserving policy
	NeverFitsRejected      int
	TotalCompletedReqs     int
	TotalTokensServed      int
	GoodputTokensPerSec    float64
	GoodputReqsPerSec      float64
	JainsIndexSliding      float64
	MaxMinRatioSliding     float64
	SlidingWindowTokens    int
	TimeWindowUs           int64
	JainsIndexDemandCapped float64
	PerTenant              map[string]*TenantMetrics
}
