package policy_test

import (
	"go/parser"
	"go/token"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"testing"

	"fairlane/policy"
)

// mockRequest implements policy.RequestView
type mockRequest struct {
	id        string
	tenantID  string
	input     int
	predOut   int
	priority  int
	arrivalUs int64
}

func (r *mockRequest) ID() string                    { return r.id }
func (r *mockRequest) TenantID() string              { return r.tenantID }
func (r *mockRequest) InputTokens() int              { return r.input }
func (r *mockRequest) PredictedOutputTokens() int     { return r.predOut }
func (r *mockRequest) ArrivalTime() int64            { return r.arrivalUs }
func (r *mockRequest) Priority() int                 { return r.priority }

// mockSchedulerView implements policy.SchedulerView
type mockSchedulerView struct {
	backlogged []string
	heads      map[string]policy.RequestView
	curTime    int64
}

func newMockView(heads map[string]policy.RequestView, curTime int64) *mockSchedulerView {
	var backlogged []string
	for t := range heads {
		backlogged = append(backlogged, t)
	}
	sort.Strings(backlogged) // Lexicographically sorted by contract
	return &mockSchedulerView{
		backlogged: backlogged,
		heads:      heads,
		curTime:    curTime,
	}
}

func (v *mockSchedulerView) BackloggedTenants() []string {
	return v.backlogged
}

func (v *mockSchedulerView) HeadRequest(tenantID string) (policy.RequestView, bool) {
	req, ok := v.heads[tenantID]
	return req, ok
}

func (v *mockSchedulerView) CurrentTime() int64 {
	return v.curTime
}

// TestNoSimImport guarantees that the policy package has zero dependency on sim.
func TestNoSimImport(t *testing.T) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", nil, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("failed to parse policy directory: %v", err)
	}

	for pkgName, pkg := range pkgs {
		for filename, file := range pkg.Files {
			for _, imp := range file.Imports {
				path := strings.Trim(imp.Path.Value, `"`)
				if strings.Contains(path, "sim") {
					t.Fatalf("VIOLATION: file %s in package %s imports forbidden package %s", filename, pkgName, path)
				}
			}
		}
	}
}

// TestRequestViewHasNoTrueOutputLength ensures the policy-visible request interface
// does not expose the true output length.
func TestRequestViewHasNoTrueOutputLength(t *testing.T) {
	reqViewType := reflect.TypeOf((*policy.RequestView)(nil)).Elem()

	for i := 0; i < reqViewType.NumMethod(); i++ {
		methodName := reqViewType.Method(i).Name
		if methodName == "OutputTokens" || methodName == "TrueOutputTokens" || methodName == "ActualOutputTokens" {
			t.Fatalf("VIOLATION: RequestView must not expose true output length! Found method: %s", methodName)
		}
	}

	// Must have PredictedOutputTokens
	_, hasPredicted := reqViewType.MethodByName("PredictedOutputTokens")
	if !hasPredicted {
		t.Fatalf("RequestView should expose PredictedOutputTokens() int")
	}
}

// TestFIFOHandComputedOracle verifies FIFO picks by earliest arrival time and tie-breaks by tenant ID.
func TestFIFOHandComputedOracle(t *testing.T) {
	fifo := policy.NewFIFO()

	reqA := &mockRequest{id: "r1", tenantID: "tenant-A", arrivalUs: 100}
	reqB := &mockRequest{id: "r2", tenantID: "tenant-B", arrivalUs: 50}
	reqC := &mockRequest{id: "r3", tenantID: "tenant-C", arrivalUs: 150}

	view := newMockView(map[string]policy.RequestView{
		"tenant-A": reqA,
		"tenant-B": reqB,
		"tenant-C": reqC,
	}, 200)

	winner := fifo.Pick(view)
	if winner != "tenant-B" {
		t.Fatalf("expected tenant-B (earliest arrival 50), got %s", winner)
	}

	// Tie-break test: both arrived at 50
	reqA.arrivalUs = 50
	viewTie := newMockView(map[string]policy.RequestView{
		"tenant-B": reqB,
		"tenant-A": reqA,
	}, 200)

	winnerTie := fifo.Pick(viewTie)
	if winnerTie != "tenant-A" {
		t.Fatalf("expected tie-break winner tenant-A, got %s", winnerTie)
	}
}

// TestRoundRobinHandComputedOracle verifies RR cycles across active tenants.
func TestRoundRobinHandComputedOracle(t *testing.T) {
	rr := policy.NewRoundRobin()

	reqA := &mockRequest{id: "r1", tenantID: "tenant-A"}
	reqB := &mockRequest{id: "r2", tenantID: "tenant-B"}
	reqC := &mockRequest{id: "r3", tenantID: "tenant-C"}

	view := newMockView(map[string]policy.RequestView{
		"tenant-A": reqA,
		"tenant-B": reqB,
		"tenant-C": reqC,
	}, 100)

	// Step 1: picks first tenant (tenant-A)
	p1 := rr.Pick(view)
	if p1 != "tenant-A" {
		t.Fatalf("expected tenant-A on pick 1, got %s", p1)
	}

	// Step 2: picks tenant-B
	p2 := rr.Pick(view)
	if p2 != "tenant-B" {
		t.Fatalf("expected tenant-B on pick 2, got %s", p2)
	}

	// Step 3: picks tenant-C
	p3 := rr.Pick(view)
	if p3 != "tenant-C" {
		t.Fatalf("expected tenant-C on pick 3, got %s", p3)
	}

	// Step 4: wraps around to tenant-A
	p4 := rr.Pick(view)
	if p4 != "tenant-A" {
		t.Fatalf("expected tenant-A on pick 4, got %s", p4)
	}
}

// TestVTCHandComputedOracle verifies VTC selects argmin(counter) and updates counters properly.
func TestVTCHandComputedOracle(t *testing.T) {
	cfg := policy.DefaultVTCConfig() // w_in=1, w_out=2
	vtc := policy.NewVTC(cfg)

	reqA := &mockRequest{id: "r1", tenantID: "tenant-A", input: 10}
	reqB := &mockRequest{id: "r2", tenantID: "tenant-B", input: 20}

	vtc.OnArrival(reqA, true, "", nil)
	vtc.OnArrival(reqB, true, "", []string{"tenant-A"})

	view := newMockView(map[string]policy.RequestView{
		"tenant-A": reqA,
		"tenant-B": reqB,
	}, 100)

	// Initially counters are both 0. Tie-break should pick tenant-A.
	w1 := vtc.Pick(view)
	if w1 != "tenant-A" {
		t.Fatalf("expected tenant-A by tie-break, got %s", w1)
	}

	// Dispatch tenant-A: cost = 1 * 10 = 10. counter_A = 10.
	vtc.OnDispatch(reqA)
	if cA := vtc.Counter("tenant-A"); cA != 10.0 {
		t.Fatalf("expected counter A=10, got %f", cA)
	}

	// Next pick: counter_A=10, counter_B=0. Must pick tenant-B!
	w2 := vtc.Pick(view)
	if w2 != "tenant-B" {
		t.Fatalf("expected tenant-B (counter 0 < 10), got %s", w2)
	}

	// Dispatch tenant-B: cost = 1 * 20 = 20. counter_B = 20.
	vtc.OnDispatch(reqB)
	if cB := vtc.Counter("tenant-B"); cB != 20.0 {
		t.Fatalf("expected counter B=20, got %f", cB)
	}

	// Next pick: counter_A=10 < counter_B=20. Must pick tenant-A!
	w3 := vtc.Pick(view)
	if w3 != "tenant-A" {
		t.Fatalf("expected tenant-A (counter 10 < 20), got %s", w3)
	}
}

// TestVTCLiftModes tests the exact difference between Paper and Artifact lift modes.
func TestVTCLiftModes(t *testing.T) {
	// Case 1: Active queue lift (both modes agree)
	// Tenant A has counter 50 and is waiting.
	// Tenant B rejoins (counter 0). B must be lifted to 50.
	for _, mode := range []policy.LiftMode{policy.LiftModePaper, policy.LiftModeArtifact} {
		cfg := policy.DefaultVTCConfig()
		cfg.LiftMode = mode
		cfg.LiftTiming = policy.LiftTimingArrival
		vtc := policy.NewVTC(cfg)

		reqA := &mockRequest{id: "r1", tenantID: "tenant-A", input: 50}
		vtc.OnArrival(reqA, true, "", nil)
		vtc.OnDispatch(reqA) // counter_A = 50

		// Tenant B arrives while Tenant A is active
		reqB := &mockRequest{id: "r2", tenantID: "tenant-B", input: 10}
		vtc.OnArrival(reqB, true, "", []string{"tenant-A"})

		if cB := vtc.Counter("tenant-B"); cB != 50.0 {
			t.Fatalf("[%s] expected Tenant B lifted to 50, got %f", mode, cB)
		}
	}

	// Case 2: System idle / Empty-queue rejoin (Paper vs Artifact difference)
	// Tenant A ran and accumulated counter 100, then queue became completely empty.
	// Now Tenant B arrives with counter 0.
	// Paper mode: B is lifted to max(0, c_A) = 100.
	// Artifact mode: B is NOT lifted; retains 0.
	{
		// Paper mode
		cfgPaper := policy.DefaultVTCConfig()
		cfgPaper.LiftMode = policy.LiftModePaper
		cfgPaper.LiftTiming = policy.LiftTimingArrival
		vtcPaper := policy.NewVTC(cfgPaper)

		reqA := &mockRequest{id: "r1", tenantID: "tenant-A", input: 100}
		vtcPaper.OnArrival(reqA, true, "", nil)
		vtcPaper.OnDispatch(reqA) // counter_A = 100

		// Queue becomes empty as A completes
		// A departed: lastDepartedTenant = "tenant-A"
		// Now system is idle (otherBacklogged = nil). B arrives:
		reqB := &mockRequest{id: "r2", tenantID: "tenant-B", input: 10}
		vtcPaper.OnArrival(reqB, true, "tenant-A", nil)

		if cB := vtcPaper.Counter("tenant-B"); cB != 100.0 {
			t.Fatalf("[Paper] expected Tenant B lifted to 100 (c_last), got %f", cB)
		}

		// Artifact mode
		cfgArt := policy.DefaultVTCConfig()
		cfgArt.LiftMode = policy.LiftModeArtifact
		cfgArt.LiftTiming = policy.LiftTimingArrival
		vtcArt := policy.NewVTC(cfgArt)

		vtcArt.OnArrival(reqA, true, "", nil)
		vtcArt.OnDispatch(reqA) // counter_A = 100

		// Queue empty, system idle (otherBacklogged = nil). B arrives:
		vtcArt.OnArrival(reqB, true, "tenant-A", nil)
		if cB := vtcArt.Counter("tenant-B"); cB != 0.0 {
			t.Fatalf("[Artifact] expected Tenant B NOT lifted on empty queue (counter 0), got %f", cB)
		}
	}
}

// TestDeterminismAgainstMapIteration ensures that map order cannot leak and cause non-deterministic decisions.
func TestDeterminismAgainstMapIteration(t *testing.T) {
	tenants := []string{"tenant-01", "tenant-02", "tenant-03", "tenant-04", "tenant-05"}

	factory := map[string]func() policy.Policy{
		"fifo":        func() policy.Policy { return policy.NewFIFO() },
		"round-robin": func() policy.Policy { return policy.NewRoundRobin() },
		"vtc":         func() policy.Policy { return policy.NewVTC(policy.DefaultVTCConfig()) },
	}

	for name, createPolicy := range factory {
		t.Run(name, func(t *testing.T) {
			var firstDecision string

			// Each iteration creates a fresh policy instance and tests that shuffled map insertion
			// produces the EXACT same decision.
			for iter := 0; iter < 50; iter++ {
				p := createPolicy()

				shuffled := append([]string(nil), tenants...)
				rand.Shuffle(len(shuffled), func(i, j int) {
					shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
				})

				heads := make(map[string]policy.RequestView)
				for _, tid := range shuffled {
					heads[tid] = &mockRequest{id: "req-" + tid, tenantID: tid, arrivalUs: 100}
				}

				view := newMockView(heads, 100)
				decision := p.Pick(view)

				if iter == 0 {
					firstDecision = decision
				} else if decision != firstDecision {
					t.Fatalf("Non-deterministic decision in policy %s! Expected %s, got %s on iteration %d",
						name, firstDecision, decision, iter)
				}
			}
		})
	}
}

func TestRoundRobinAdvanceOnPickVsOnDispatch(t *testing.T) {
	heads := map[string]policy.RequestView{
		"chat": &mockRequest{id: "chat-1", tenantID: "chat", arrivalUs: 100},
		"rag":  &mockRequest{id: "rag-1", tenantID: "rag", arrivalUs: 100},
	}
	view := newMockView(heads, 100)

	// 1. Advance OnPick
	rrPick := policy.NewRoundRobinWithAdvance(policy.RRAdvanceOnPick)
	first := rrPick.Pick(view)
	if first != "chat" {
		t.Fatalf("Expected chat as first pick, got %s", first)
	}
	second := rrPick.Pick(view)
	if second != "rag" {
		t.Fatalf("Expected cursor to advance on pick to rag, got %s", second)
	}

	// 2. Advance OnDispatch
	rrDisp := policy.NewRoundRobinWithAdvance(policy.RRAdvanceOnDispatch)
	p1 := rrDisp.Pick(view)
	if p1 != "chat" {
		t.Fatalf("Expected chat as first pick, got %s", p1)
	}
	// Pick again WITHOUT calling OnDispatch
	p2 := rrDisp.Pick(view)
	if p2 != "chat" {
		t.Fatalf("Expected cursor NOT to advance on failed dispatch, got %s (expected chat)", p2)
	}
	// Now call OnDispatch for chat
	rrDisp.OnDispatch(&mockRequest{id: "chat-1", tenantID: "chat", arrivalUs: 100})
	p3 := rrDisp.Pick(view)
	if p3 != "rag" {
		t.Fatalf("Expected cursor to advance to rag after OnDispatch, got %s", p3)
	}
}

func TestVTCLiftModeNone(t *testing.T) {
	// 1. LiftModeArtifact: counter IS lifted to min of other active tenants
	cfgArt := policy.DefaultVTCConfig()
	cfgArt.LiftMode = policy.LiftModeArtifact
	vtcArt := policy.NewVTC(cfgArt)

	// Tenant A accumulates 5000 tokens
	vtcArt.OnDispatch(&mockRequest{id: "a1", tenantID: "tenant-a", input: 5000})

	// Tenant B arrives while Tenant A is backlogged
	reqB := &mockRequest{id: "b1", tenantID: "tenant-b", input: 100}
	vtcArt.OnArrival(reqB, true, "", []string{"tenant-a"})

	if vtcArt.Counter("tenant-b") != 5000 {
		t.Fatalf("Expected tenant-b counter lifted to 5000 under LiftModeArtifact, got %.1f", vtcArt.Counter("tenant-b"))
	}

	// 2. LiftModeNone: counter is NEVER lifted
	cfgNone := policy.DefaultVTCConfig()
	cfgNone.LiftMode = policy.LiftModeNone
	vtcNone := policy.NewVTC(cfgNone)

	vtcNone.OnDispatch(&mockRequest{id: "a1", tenantID: "tenant-a", input: 5000})
	vtcNone.OnArrival(reqB, true, "", []string{"tenant-a"})

	if vtcNone.Counter("tenant-b") != 0 {
		t.Fatalf("Expected tenant-b counter to remain 0 under LiftModeNone, got %.1f", vtcNone.Counter("tenant-b"))
	}
}

