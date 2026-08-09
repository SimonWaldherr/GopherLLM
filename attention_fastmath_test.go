package gopherllm

import (
	"math"
	"testing"
)

// attentionWeightsInPlace is shared by the grouped-GQA path. Keep its f32
// transcendental implementation pinned to the former float64-libm result so
// an optimisation cannot silently perturb a softmax distribution.
func TestAttentionWeightsFastMathMatchesLibm(t *testing.T) {
	for _, tc := range []struct {
		name    string
		scores  []float32
		softcap float32
	}{
		{"ordinary", []float32{-9, -3.5, -0.25, 0, 0.7, 3, 9}, 0},
		{"longTail", []float32{-80, -40, -20, -5, -1, 0}, 0},
		{"softcap", []float32{-50, -9, -2, 0, 1.5, 8, 50}, 2.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := append([]float32(nil), tc.scores...)
			gotDenom := attentionWeightsInPlace(got, tc.softcap)

			wantScores := append([]float32(nil), tc.scores...)
			if tc.softcap > 0 {
				for i, s := range wantScores {
					wantScores[i] = tc.softcap * float32(math.Tanh(float64(s/tc.softcap)))
				}
			}
			maxScore := wantScores[0]
			for _, s := range wantScores[1:] {
				if s > maxScore {
					maxScore = s
				}
			}
			var wantDenom float32
			for i, s := range wantScores {
				wantScores[i] = float32(math.Exp(float64(s - maxScore)))
				wantDenom += wantScores[i]
			}

			if diff := math.Abs(float64(gotDenom - wantDenom)); diff > 2e-5*math.Max(1, math.Abs(float64(wantDenom))) {
				t.Fatalf("denom = %g, want %g (diff %g)", gotDenom, wantDenom, diff)
			}
			for i := range got {
				if diff := math.Abs(float64(got[i] - wantScores[i])); diff > 2e-6 {
					t.Fatalf("weight[%d] = %g, want %g (diff %g)", i, got[i], wantScores[i], diff)
				}
			}
		})
	}
}
