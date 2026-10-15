package sim

import "fairlane/policy"

// TimeoutCorrectionWrapper intercepts policy callbacks and drops OnComplete for timed-out/cancelled requests.
type TimeoutCorrectionWrapper struct {
	policy.Policy
	CancelledIDs       map[string]bool
	TimeoutsFired      int
	DroppedCorrections int
}

// NewTimeoutCorrectionWrapper wraps a policy to drop completion callbacks for cancelled request IDs.
func NewTimeoutCorrectionWrapper(pol policy.Policy, cancelledIDs map[string]bool) *TimeoutCorrectionWrapper {
	if cancelledIDs == nil {
		cancelledIDs = make(map[string]bool)
	}
	return &TimeoutCorrectionWrapper{
		Policy:       pol,
		CancelledIDs: cancelledIDs,
	}
}

func (w *TimeoutCorrectionWrapper) OnComplete(tenantID string, req policy.RequestView, in, out int) {
	if req != nil && w.CancelledIDs[req.ID()] {
		w.TimeoutsFired++
		w.DroppedCorrections++
		return
	}
	w.Policy.OnComplete(tenantID, req, in, out)
}

// NeverCorrectWrapper completely disables completion corrections.
type NeverCorrectWrapper struct {
	policy.Policy
}

func NewNeverCorrectWrapper(pol policy.Policy) *NeverCorrectWrapper {
	return &NeverCorrectWrapper{Policy: pol}
}

func (w *NeverCorrectWrapper) OnComplete(tenantID string, req policy.RequestView, in, out int) {}
