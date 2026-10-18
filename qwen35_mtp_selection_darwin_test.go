//go:build darwin && cgo && metal

package gopherllm

import (
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

// These handles are only used to inspect selection, never dispatched or
// released. No GPU/device access is needed to test the routing contract.
func mtpSelectionWeight(typ GGMLType, rows, cols int) Weight {
	m := &MetalWeight{typ: typ, rows: rows, cols: cols}
	switch typ {
	case GGMLTypeQ4_K:
		m.q4 = new(metalbackend.Weight)
	case GGMLTypeQ6_K:
		m.q6 = new(metalbackend.Weight)
	}
	return Weight{Metal: m, Type: typ, Rows: rows, Cols: cols}
}

func mtpSelectionFixture() Qwen35MTPWeights {
	return Qwen35MTPWeights{
		TokenEmbd: mtpSelectionWeight(GGMLTypeQ6_K, 32768, 3072), EHProj: mtpSelectionWeight(GGMLTypeQ4_K, 3072, 6144),
		Attention: Qwen35AttentionWeights{Q: mtpSelectionWeight(GGMLTypeQ4_K, 8192, 3072),
			K: mtpSelectionWeight(GGMLTypeQ6_K, 4096, 3072), V: mtpSelectionWeight(GGMLTypeQ6_K, 4096, 3072),
			O: mtpSelectionWeight(GGMLTypeQ6_K, 3072, 2048)},
		FFN: Qwen35FFNWeights{Gate: mtpSelectionWeight(GGMLTypeQ4_K, 8192, 3072),
			Up: mtpSelectionWeight(GGMLTypeQ4_K, 8192, 3072), Down: mtpSelectionWeight(GGMLTypeQ6_K, 3072, 8192)},
		Output: mtpSelectionWeight(GGMLTypeQ6_K, 32768, 3072),
	}
}

func TestQwen35MTPExecutionWeightsKeepsCPUKVAndMetalOutput(t *testing.T) {
	oldBody, oldFused := mtpMetalBodyEnabled, metalFusedFFNEnabled
	mtpMetalBodyEnabled, metalFusedFFNEnabled = true, true
	t.Cleanup(func() { mtpMetalBodyEnabled, metalFusedFFNEnabled = oldBody, oldFused })
	original := mtpSelectionFixture()
	before := original
	handles := [...]*MetalWeight{original.TokenEmbd.Metal, original.EHProj.Metal, original.Attention.Q.Metal,
		original.Attention.K.Metal, original.Attention.V.Metal, original.Attention.O.Metal,
		original.FFN.Gate.Metal, original.FFN.Up.Metal, original.FFN.Down.Metal, original.Output.Metal}
	descriptors := make([]MetalWeight, len(handles))
	for i, handle := range handles {
		descriptors[i] = *handle
	}
	selected := qwen35MTPExecutionWeights(original)
	if selected.TokenEmbd.Metal != nil || selected.EHProj.Metal != nil || selected.Attention.K.Metal != nil || selected.Attention.V.Metal != nil {
		t.Fatal("MTP input-pair or K/V projections left the established CPU path")
	}
	if selected.Attention.Q.Metal != original.Attention.Q.Metal || selected.Attention.O.Metal != original.Attention.O.Metal ||
		selected.FFN.Gate.Metal != original.FFN.Gate.Metal || selected.FFN.Up.Metal != original.FFN.Up.Metal || selected.FFN.Down.Metal != original.FFN.Down.Metal {
		t.Fatal("eligible Q/O or complete fused FFN lost its Metal view")
	}
	if selected.Output.Metal != original.Output.Metal {
		t.Fatal("MTP body selection changed the independent vocabulary head")
	}
	if original.TokenEmbd.Metal != before.TokenEmbd.Metal || original.EHProj.Metal != before.EHProj.Metal ||
		original.Attention.Q.Metal != before.Attention.Q.Metal || original.Attention.K.Metal != before.Attention.K.Metal ||
		original.Attention.V.Metal != before.Attention.V.Metal || original.Attention.O.Metal != before.Attention.O.Metal ||
		original.FFN.Gate.Metal != before.FFN.Gate.Metal || original.FFN.Up.Metal != before.FFN.Up.Metal ||
		original.FFN.Down.Metal != before.FFN.Down.Metal || original.Output.Metal != before.Output.Metal {
		t.Fatal("selecting an execution view mutated the original weight ownership")
	}
	for i, handle := range handles {
		if *handle != descriptors[i] {
			t.Fatal("MTP selection mutated an original Metal descriptor")
		}
	}
	mtpMetalBodyEnabled = false
	disabled := qwen35MTPExecutionWeights(original)
	want := mtpHeadCPUWeights(original)
	if disabled.TokenEmbd.Metal != nil || disabled.EHProj.Metal != nil || disabled.Attention.Q.Metal != nil || disabled.Attention.K.Metal != nil ||
		disabled.Attention.V.Metal != nil || disabled.Attention.O.Metal != nil || disabled.FFN.Gate.Metal != nil || disabled.FFN.Up.Metal != nil || disabled.FFN.Down.Metal != nil || disabled.Output.Metal != want.Output.Metal {
		t.Fatal("MTP Metal-body opt-out did not restore CPU body views while retaining the output head")
	}
}

func TestQwen35MTPExecutionWeightsRejectsPartialFFNAndNarrowDirect(t *testing.T) {
	oldBody, oldFused := mtpMetalBodyEnabled, metalFusedFFNEnabled
	mtpMetalBodyEnabled, metalFusedFFNEnabled = true, true
	t.Cleanup(func() { mtpMetalBodyEnabled, metalFusedFFNEnabled = oldBody, oldFused })
	for _, broken := range []string{"missing up", "shape mismatch", "fusion disabled"} {
		t.Run(broken, func(t *testing.T) {
			w := mtpSelectionFixture()
			w.Attention.Q = mtpSelectionWeight(GGMLTypeQ4_K, 3072, 8192)
			w.Attention.O = mtpSelectionWeight(GGMLTypeQ4_K, 3072, 8192)
			metalFusedFFNEnabled = true
			switch broken {
			case "missing up":
				w.FFN.Up.Metal = nil
			case "shape mismatch":
				w.FFN.Up.Metal.rows++
			case "fusion disabled":
				metalFusedFFNEnabled = false
			}
			selected := qwen35MTPExecutionWeights(w)
			if selected.Attention.Q.Metal != nil || selected.Attention.O.Metal != nil {
				t.Fatal("fusion-only narrow Q/O handles were selected for standalone matvecs")
			}
			if selected.FFN.Gate.Metal != nil || selected.FFN.Up.Metal != nil || selected.FFN.Down.Metal != nil {
				t.Fatal("partial or disabled fused FFN retained a mixed body route")
			}
			if selected.Output.Metal != w.Output.Metal {
				t.Fatal("FFN fallback changed the vocabulary head")
			}
		})
	}
}
