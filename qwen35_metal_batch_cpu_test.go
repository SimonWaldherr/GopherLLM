package gopherllm

import "testing"

func TestQwen35MetalBatchCPUOutRequiresQ8BlockWidth(t *testing.T) {
	previousQ8 := useQ8Activations.Load()
	useQ8Activations.Store(true)
	t.Cleanup(func() { useQ8Activations.Store(previousQ8) })
	for _, tc := range []struct {
		name       string
		rows, cols int
		want       bool
	}{
		{"supported", 8, 256, true},
		// A valid Q8_0 matrix can have only four 32-element blocks. That
		// width cannot use the shared Q8_K activation kernel, so batching it
		// would fall through to a different floating-point operation tree.
		{"Q8_0 four blocks", 8, 128, false},
		{"empty rows", 0, 256, false},
		{"empty columns", 8, 0, false},
		{"negative rows", -1, 256, false},
		{"negative columns", 8, -256, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := Weight{Type: GGMLTypeQ8_0, Rows: tc.rows, Cols: tc.cols, Raw: make([]byte, 8*256*34/32)}
			if got := qwen35MetalBatchCPUOut(w); got != tc.want {
				t.Fatalf("CPU-Q8 output eligibility=%t, want %t for %dx%d", got, tc.want, tc.rows, tc.cols)
			}
		})
	}
}
