package laya_test

import (
	"context"
	"os"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// GOPHERLLM_LAYA_BENCH_MODEL selects real weights; loading is outside timing.
func BenchmarkLayaPredict(b *testing.B) {
	dir := os.Getenv("GOPHERLLM_LAYA_BENCH_MODEL")
	if dir == "" {
		dir = "../../testdata/laya-tiny"
	}
	m, err := gopherllm.OpenLaya(context.Background(), dir)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	gopherllm.SetNumThreads(4)
	for _, tc := range []struct {
		name   string
		repeat int
	}{{"short", 1}, {"long", 30}} {
		b.Run(tc.name, func(b *testing.B) {
			req := gopherllm.DecisionRequest{State: strings.Repeat("Please refund my payment. The application does not work. ", tc.repeat), Questions: map[string]gopherllm.DecisionQuestion{"result": {Type: "choice", Instructions: "Which department should handle this request?", Criteria: []string{"billing", "technical support", "sales"}}}}
			if _, err = m.Predict(context.Background(), req); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := m.Predict(context.Background(), req); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
