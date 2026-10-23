//go:build darwin && cgo && metal

package gopherllm

import (
	"bufio"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// Use actual crossover-sized projections and a complete contracting FFN,
// with rectangular input/output widths to bound the mapped fixture to 18 MB.
// The test exercises resource ownership and projection routing, not a whole
// draft transformer with checkpoint-sized activations.
func qwen35MappedMTPFixture(t *testing.T) (Qwen35MTPWeights, *MmapFile) {
	t.Helper()
	file, err := os.Create(filepath.Join(t.TempDir(), "mtp-weights.bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	writer := bufio.NewWriterSize(file, 64<<10)
	if _, err := writer.Write(make([]byte, 32)); err != nil {
		t.Fatal(err)
	}
	type descriptor struct {
		typ                GGMLType
		rows, cols, offset int
	}
	descriptors := []descriptor{
		{typ: GGMLTypeQ6_K, rows: 3072, cols: 256},  // Q
		{typ: GGMLTypeQ6_K, rows: 3072, cols: 256},  // O
		{typ: GGMLTypeQ4_K, rows: 8192, cols: 256},  // gate
		{typ: GGMLTypeQ4_K, rows: 8192, cols: 256},  // up
		{typ: GGMLTypeQ6_K, rows: 2048, cols: 8192}, // contracting down
		{typ: GGMLTypeQ6_K, rows: 3072, cols: 256},  // vocabulary head
	}
	offset := 32
	for i := range descriptors {
		d := &descriptors[i]
		d.offset = offset
		rng := rand.New(rand.NewSource(int64(170 + i)))
		row := randomQ6KRow(rng, d.cols)
		if d.typ == GGMLTypeQ4_K {
			row = randomQ4KRow(rng, d.cols)
		}
		for range d.rows {
			if _, err := writer.Write(row); err != nil {
				t.Fatal(err)
			}
		}
		offset += d.rows * len(row)
	}
	if err := writer.Flush(); err != nil {
		t.Fatal(err)
	}
	mapped, err := OpenMmap(file.Name())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mapped.Close() })
	if !mapped.IsMapped() || mapped.Len() > 30<<20 {
		t.Fatalf("fixture is not a bounded OS mapping: mapped=%t bytes=%d", mapped.IsMapped(), mapped.Len())
	}
	weights := make([]Weight, len(descriptors))
	for i, d := range descriptors {
		end := mapped.Len()
		if i+1 < len(descriptors) {
			end = descriptors[i+1].offset
		}
		weights[i] = Weight{Raw: mapped.Bytes()[d.offset:end], Type: d.typ, Rows: d.rows, Cols: d.cols}
	}
	w := Qwen35MTPWeights{
		TokenEmbd: weights[5], EHProj: weights[0],
		Attention: Qwen35AttentionWeights{Q: weights[0], K: weights[0], V: weights[1], O: weights[1]},
		FFN:       Qwen35FFNWeights{Gate: weights[2], Up: weights[3], Down: weights[4]}, Output: weights[5],
		lazyMetal: &qwen35MTPLazyMetal{mapping: mapped.Bytes()},
	}
	t.Cleanup(func() { releaseQwen35MetalWeights(&Qwen35Weights{MTP: &w}) })
	return w, mapped
}

func qwen35LazyMTPBodySetting(t *testing.T, enabled bool) {
	t.Helper()
	oldBody, oldFused := mtpMetalBodyEnabled, metalFusedFFNEnabled
	mtpMetalBodyEnabled, metalFusedFFNEnabled = enabled, true
	t.Cleanup(func() { mtpMetalBodyEnabled, metalFusedFFNEnabled = oldBody, oldFused })
}

func qwen35LazyMTPHandles(w Qwen35MTPWeights) [6]*MetalWeight {
	return [6]*MetalWeight{w.Attention.Q.Metal, w.Attention.O.Metal,
		w.FFN.Gate.Metal, w.FFN.Up.Metal, w.FFN.Down.Metal, w.Output.Metal}
}

func requireQwen35LazyMTPDormant(t *testing.T, w Qwen35MTPWeights) {
	t.Helper()
	if w.lazyMetal == nil {
		t.Fatal("MTP weights lost deferred Metal registration")
	}
	for _, handle := range qwen35LazyMTPHandles(w) {
		if handle != nil {
			t.Fatal("unused MTP projection was prepared eagerly")
		}
	}
	lazy := w.lazyMetal
	if lazy.q != nil || lazy.o != nil || lazy.gate != nil || lazy.up != nil || lazy.down != nil || lazy.output != nil {
		t.Fatal("unused MTP head registered hidden Metal handles")
	}
}

func requireQwen35MTPMetalReleased(t *testing.T, handles [6]*MetalWeight) {
	t.Helper()
	for _, handle := range handles {
		if handle != nil && (handle.q4 != nil || handle.q5 != nil || handle.q6 != nil || handle.q8 != nil) {
			t.Fatal("closing MTP retained a Metal backend handle")
		}
	}
}

func TestQwen35LazyMTPParallelInitializationAndEagerParity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	qwen35LazyMTPBodySetting(t, true)
	w, _ := qwen35MappedMTPFixture(t)
	requireQwen35LazyMTPDormant(t, w)
	const callers = 16
	views := make([]Qwen35MTPWeights, callers)
	var wait sync.WaitGroup
	for i := range views {
		wait.Add(1)
		go func() {
			defer wait.Done()
			views[i] = qwen35MTPExecutionWeights(w)
		}()
	}
	wait.Wait()
	handles := qwen35LazyMTPHandles(views[0])
	for _, handle := range handles {
		if handle == nil {
			t.Fatalf("eligible lazy projection is unavailable: %s", MetalError())
		}
	}
	for _, view := range views {
		if view.lazyMetal != nil || qwen35LazyMTPHandles(view) != handles {
			t.Fatal("concurrent execution views did not share one initialization")
		}
		if view.TokenEmbd.Metal != nil || view.EHProj.Metal != nil || view.Attention.K.Metal != nil || view.Attention.V.Metal != nil {
			t.Fatal("deferred initialization offloaded CPU-only MTP projections")
		}
	}
	for _, handle := range qwen35LazyMTPHandles(w) {
		if handle != nil {
			t.Fatal("execution-view selection mutated original ownership descriptors")
		}
	}
	if w.lazyMetal == nil {
		t.Fatal("original MTP owner lost its deferred resources")
	}
	eager := w
	eager.lazyMetal = nil
	for _, weight := range []*Weight{&eager.Attention.Q, &eager.Attention.O,
		&eager.FFN.Gate, &eager.FFN.Up, &eager.FFN.Down, &eager.Output} {
		weight.Metal = prepareMetalWeight(weight.Raw, weight.Type, weight.Rows, weight.Cols, true)
		if weight.Metal == nil {
			t.Fatalf("prepare eager comparison: %s", MetalError())
		}
	}
	t.Cleanup(func() { releaseQwen35MetalWeights(&Qwen35Weights{MTP: &eager}) })
	selected := views[0]
	for _, pair := range [][2]Weight{{selected.Attention.Q, eager.Attention.Q},
		{selected.Attention.O, eager.Attention.O}, {selected.Output, eager.Output}} {
		x := metalTestVector(pair[0].Cols)
		var actual, expected []float32
		pair[0].MatvecInto(x, &actual)
		pair[1].MatvecInto(x, &expected)
		requireQwen35LazyMTPValues(t, actual, expected)
	}
	var actual, expected []float32
	x := metalTestVector(selected.FFN.Gate.Cols)
	if !matvecMetalQwen35SwiGLUInto(selected.FFN.Gate.Metal, selected.FFN.Up.Metal, selected.FFN.Down.Metal, x, &actual) ||
		!matvecMetalQwen35SwiGLUInto(eager.FFN.Gate.Metal, eager.FFN.Up.Metal, eager.FFN.Down.Metal, x, &expected) {
		t.Fatalf("complete fused FFN was not selected: %s", MetalError())
	}
	requireQwen35LazyMTPValues(t, actual, expected)
	owner := Qwen35Weights{MTP: &w}
	releaseQwen35MetalWeights(&owner)
	if w.lazyMetal != nil {
		t.Fatal("release retained lazy MTP ownership")
	}
	requireQwen35MTPMetalReleased(t, handles)
	releaseQwen35MetalWeights(&owner)
}

func requireQwen35LazyMTPValues(t *testing.T, actual, expected []float32) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("projection widths differ: %d/%d", len(actual), len(expected))
	}
	for i := range actual {
		if math.IsNaN(float64(actual[i])) || math.IsInf(float64(actual[i]), 0) || math.Float32bits(actual[i]) != math.Float32bits(expected[i]) {
			t.Fatalf("projection row %d differs: %g/%g", i, actual[i], expected[i])
		}
	}
}

func TestQwen35LazyMTPBodyOptOutAndSharedOutputClose(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	qwen35LazyMTPBodySetting(t, false)
	w, mapped := qwen35MappedMTPFixture(t)
	w.Output.Metal = prepareMetalWeight(w.Output.Raw, w.Output.Type, w.Output.Rows, w.Output.Cols, true)
	if w.Output.Metal == nil {
		t.Fatalf("prepare shared trunk head: %s", MetalError())
	}
	shared := w.Output.Metal
	runner := &Runner{kind: loadedQwen35, qwen35: Qwen35Weights{Output: w.Output, MTP: &w}, mappedFile: mapped}
	t.Cleanup(func() { _ = runner.Close() })
	selected := qwen35MTPExecutionWeights(w)
	if selected.Output.Metal != shared || w.lazyMetal.output != shared {
		t.Fatal("lazy MTP duplicated the existing shared vocabulary head")
	}
	selectedHandles := qwen35LazyMTPHandles(selected)
	for _, handle := range selectedHandles[:5] {
		if handle != nil {
			t.Fatal("MTP body opt-out selected Metal")
		}
	}
	lazy := w.lazyMetal
	if lazy.q != nil || lazy.o != nil || lazy.gate != nil || lazy.up != nil || lazy.down != nil {
		t.Fatal("body opt-out registered unused body pages with Metal")
	}
	mtpMetalBodyEnabled = true
	selected = qwen35MTPExecutionWeights(w)
	handles := qwen35LazyMTPHandles(selected)
	for _, handle := range handles {
		if handle == nil {
			t.Fatalf("body was not initialized after opt-in: %s", MetalError())
		}
	}
	if selected.Output.Metal != shared {
		t.Fatal("enabling the body replaced the shared output head")
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
	if w.lazyMetal != nil || runner.qwen35.MTP != nil || runner.mappedFile != nil {
		t.Fatal("Close retained draft ownership or the mapped file")
	}
	requireQwen35MTPMetalReleased(t, handles)
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestQwen35LazyMTPRejectsOwnedAndPaddedRawViews(t *testing.T) {
	w, _ := qwen35MappedMTPFixture(t)
	for _, kind := range []string{"owned", "padded", "prepared", "float32", "truncated"} {
		t.Run(kind, func(t *testing.T) {
			weight := w.Attention.Q
			switch kind {
			case "owned", "padded":
				weight.Raw = append([]byte(nil), weight.Raw...)
				if kind == "padded" {
					weight.Raw = append(weight.Raw, make([]byte, 64)...)
				}
			case "prepared":
				weight.Prepared = &PreparedQuantizedWeight{}
			case "float32":
				weight.F32 = []float32{1}
			case "truncated":
				weight.Raw = weight.Raw[:len(weight.Raw)-1]
			}
			if handle := w.lazyMetal.prepare(weight); handle != nil {
				releaseMetalWeight(handle)
				t.Fatalf("%s raw storage escaped to Metal bytesNoCopy", kind)
			}
		})
	}
}

func TestQwen35LazyMTPRejectsPartialFFNAndNarrowProjections(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	qwen35LazyMTPBodySetting(t, true)
	w, _ := qwen35MappedMTPFixture(t)
	// Padding/copying one FFN projection must not produce a mixed GPU/CPU
	// body. Narrow Q4_K attention output remains on the CPU as before.
	w.FFN.Up.Raw = append([]byte(nil), w.FFN.Up.Raw...)
	w.Attention.O = Weight{Type: GGMLTypeQ4_K, Rows: 3072, Cols: 256,
		Raw: w.FFN.Gate.Raw[:3072*144]}
	selected := qwen35MTPExecutionWeights(w)
	if selected.Attention.Q.Metal == nil || selected.Attention.O.Metal != nil {
		t.Fatal("lazy Q/O selection ignored the standalone Metal crossover")
	}
	if selected.FFN.Gate.Metal != nil || selected.FFN.Up.Metal != nil || selected.FFN.Down.Metal != nil {
		t.Fatal("partial lazy FFN retained mixed execution backends")
	}
	if selected.Output.Metal == nil {
		t.Fatal("partial FFN disabled the independent output projection")
	}
}

func TestQwen35MappedMTPStaysDormantDuringSerialGeneration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen-mtp.gguf")
	if err := os.WriteFile(path, buildTinyQwen35MoEGGUF(true), 0o600); err != nil {
		t.Fatal(err)
	}
	runner, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, Prefault: MmapPrefaultNone})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = runner.Close() })
	if !runner.IsMapped() || runner.qwen35.MTP == nil {
		t.Fatal("tiny MTP model was not loaded through mmap")
	}
	requireQwen35LazyMTPDormant(t, *runner.qwen35.MTP)
	lazy := runner.qwen35.MTP.lazyMetal
	options := DefaultGenerationOptions()
	options.SystemPrompt, options.MaxTokens, options.MTPDraftTokens = "", 3, 0
	options.Seed, options.Sampler.Temperature, options.Sampler.TopK = 43, 0, 1
	if _, err := runner.GenerateChat([]ChatMessage{UserMessage("ab")}, options); err != nil {
		t.Fatal(err)
	}
	if runner.qwen35.MTP.lazyMetal != lazy {
		t.Fatal("serial generation replaced the dormant MTP owner")
	}
	requireQwen35LazyMTPDormant(t, *runner.qwen35.MTP)
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
	if runner.qwen35.MTP != nil || runner.mappedFile != nil {
		t.Fatal("closing unused MTP retained model resources")
	}
}
