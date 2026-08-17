package gopherllm

import "testing"

func TestArchGraphGroupsHybridLayerSchedule(t *testing.T) {
	g, err := ParseGGUFQuiet(buildTinyQwen35MoEGGUF(false))
	if err != nil {
		t.Fatal(err)
	}
	graph := BuildArchGraph(g, ConfigFromGGUF(g), AnalyzeGGUF(g, nil), nil)
	if graph.Uniform {
		t.Fatal("hybrid Qwen3.5 schedule must not be marked uniform")
	}
	if len(graph.Groups) != 2 {
		t.Fatalf("groups = %+v, want two layer groups", graph.Groups)
	}
	if got := graph.Groups[0]; got.Start != 0 || got.End != 0 || got.Attention != "deltanet" || got.FFN != "moe" {
		t.Fatalf("first group = %+v", got)
	}
	if got := graph.Groups[1]; got.Start != 1 || got.End != 1 || got.Attention != "attn" || got.FFN != "moe" || !got.QKNorm {
		t.Fatalf("second group = %+v", got)
	}
	if graph.Summary.KVCacheLayers != 1 || graph.Summary.KVCacheBytesAtFullContextF16 == 0 {
		t.Fatalf("memory summary = %+v", graph.Summary)
	}
}
