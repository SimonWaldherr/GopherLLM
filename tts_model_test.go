package gopherllm

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestTTSInverseSplineReference(t *testing.T) {
	// PyTorch 2.8 / Transformers 4.57.1, VITS inverse spline. Nonuniform
	// bins exercise interval search, edge derivatives and identity tails.
	x := []float32{-7, -5, -4.9, -3.2, -.5, 0, 1.2, 4.9, 5, 8}
	want := []float32{-7, -5, -4.9639277, -4.8150268, -4.221838, -4.006652, -3.4382007, 4.7894974, 5, 8}
	w, h, d := make([]float32, 10), make([]float32, 10), make([]float32, 9)
	for i := range w {
		w[i] = -1.1 + float32(i)*2.4/9
		h[i] = .7 - float32(i)*1.6/9
	}
	for i := range d {
		d[i] = -1 + float32(i)*2/8
	}
	for i, v := range x {
		got, err := ttsInverseSpline(v, w, h, d, 5)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(got-want[i])) > 3e-6 {
			t.Errorf("x=%g: got %g want %g", v, got, want[i])
		}
	}
	if _, err := ttsInverseSpline(float32(math.NaN()), w, h, d, 5); err == nil {
		t.Fatal("accepted NaN")
	}
}

func TestTTSTokenizerGerman(t *testing.T) {
	var vocab map[string]int
	if err := json.Unmarshal([]byte(`{" ":0,"-":5,"0":44,"1":7,"2":2,"3":13,"4":11,"5":31,"6":30,"7":42,"8":8,"_":27,"a":9,"b":22,"c":33,"d":21,"e":38,"f":6,"g":4,"h":10,"i":34,"j":28,"k":37,"l":17,"m":15,"n":18,"o":25,"p":36,"q":3,"r":14,"s":29,"t":19,"u":26,"v":1,"w":40,"x":43,"y":23,"z":41,"ß":24,"ä":16,"ë":20,"ï":32,"ö":12,"ü":35,"–":39}`), &vocab); err != nil {
		t.Fatal(err)
	}
	m := &TTSModel{vocab: make(map[rune]int)}
	for r, id := range vocab {
		m.vocab[[]rune(r)[0]] = id
	}
	for _, tc := range []struct {
		text string
		ids  []int
	}{
		{"Hallo Welt.", []int{0, 10, 0, 9, 0, 17, 0, 17, 0, 25, 0, 0, 0, 40, 0, 38, 0, 17, 0, 19, 0}},
		{" ÄÖÜ, Straße! ", []int{0, 16, 0, 12, 0, 35, 0, 0, 0, 29, 0, 19, 0, 14, 0, 9, 0, 24, 0, 38, 0}},
		{"Guten\nTag", []int{0, 4, 0, 26, 0, 19, 0, 38, 0, 18, 0, 19, 0, 9, 0, 4, 0}},
	} {
		got, err := m.tokenize(tc.text)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, tc.ids) {
			t.Errorf("%q: got %v want %v", tc.text, got, tc.ids)
		}
	}
	for _, s := range []string{"", "!?🙂", strings.Repeat("a", 513), string([]byte{255})} {
		if _, err := m.tokenize(s); err == nil {
			t.Errorf("accepted %q", s)
		}
	}
}

func TestTTSDurationsBounded(t *testing.T) {
	got, n, err := ttsDurations([]float32{0, float32(math.Log(2)), float32(math.Log(3))}, 1)
	if err != nil || n != 6 || !slices.Equal(got, []int{1, 2, 3}) {
		t.Fatalf("durations %v n=%d err=%v", got, n, err)
	}
	for _, v := range []float32{100, float32(math.Inf(1)), float32(math.NaN())} {
		if _, _, err := ttsDurations([]float32{v}, 1); err == nil {
			t.Fatalf("accepted %g", v)
		}
	}
}

func TestTTSLifetimeAndMetadata(t *testing.T) {
	var zero TTSModel
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := zero.Synthesize(context.Background(), "Hallo", TTSOptions{}); err == nil {
		t.Fatal("zero model accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := OpenTTS(ctx, t.TempDir()); err != context.Canceled {
		t.Fatalf("cancel: %v", err)
	}
	m := &TTSModel{gate: make(chan struct{}, 1)}
	m.gate <- struct{}{}
	if _, err := m.Synthesize(ctx, "Hallo", TTSOptions{}); err != context.Canceled {
		t.Fatalf("waiting cancel: %v", err)
	}
	<-m.gate
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Synthesize(context.Background(), "Hallo", TTSOptions{}); err == nil {
		t.Fatal("closed model accepted")
	}
	p := filepath.Join(t.TempDir(), "config.json")
	for _, bad := range []string{`{} {}`, strings.Repeat(" ", (1<<20)+1)} {
		if err := os.WriteFile(p, []byte(bad), 0600); err != nil {
			t.Fatal(err)
		}
		var c ttsVITSConfig
		if err := ttsReadJSON(p, &c); err == nil {
			t.Fatal("accepted invalid metadata")
		}
	}
}

func TestTTSMMSReference(t *testing.T) {
	path, reference := os.Getenv("GOPHERLLM_TTS_MODEL"), os.Getenv("GOPHERLLM_TTS_REFERENCE")
	if path == "" {
		t.Skip("set GOPHERLLM_TTS_MODEL for official MMS parity")
	}
	if reference == "" {
		reference = "testdata/tts_mms.safetensors"
	}
	ctx := context.Background()
	m, err := OpenTTS(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	var weightBytes int
	for _, w := range m.weights {
		weightBytes += 4 * len(w.data)
	}
	t.Logf("resident inference weights: %d tensors, %d bytes", len(m.weights), weightBytes)
	data, err := os.ReadFile(reference)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := ParseSafetensors(data)
	if err != nil {
		t.Fatal(err)
	}
	get := func(name string) []float32 {
		t.Helper()
		v, e := ref.F32(name)
		if e != nil {
			t.Fatal(e)
		}
		return v
	}
	check := func(name string, got []float32, tol float64) {
		t.Helper()
		want := get(name)
		if len(got) != len(want) {
			t.Fatalf("%s length %d want %d", name, len(got), len(want))
		}
		var max, sq, power float64
		for i, v := range got {
			d := float64(v - want[i])
			if math.Abs(d) > max {
				max = math.Abs(d)
			}
			sq += d * d
			power += float64(want[i]) * float64(want[i])
		}
		rel := math.Sqrt(sq / math.Max(power, 1e-30))
		t.Logf("%s: max_abs=%g relative_rms=%g", name, max, rel)
		if math.IsNaN(max) || math.IsNaN(rel) || max > tol {
			t.Errorf("%s error exceeds %g", name, tol)
		}
	}
	ids, err := m.tokenize("Hallo Welt.")
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := get("ids")
	if len(ids) != len(wantIDs) {
		t.Fatal("tokenizer length")
	}
	for i, id := range ids {
		if id != int(wantIDs[i]) {
			t.Fatalf("tokenizer[%d]=%d want %g", i, id, wantIDs[i])
		}
	}
	h, means, logs, err := m.encode(ctx, ids)
	if err != nil {
		t.Fatal(err)
	}
	check("hidden", h.data, 2e-4)
	check("means", means.data, 2e-4)
	check("logs", logs.data, 2e-4)
	d, err := m.duration(ctx, h, get("duration_noise"))
	if err != nil {
		t.Fatal(err)
	}
	check("log_duration", d, 2e-4)
	durations, frames, err := ttsDurations(d, 1)
	if err != nil {
		t.Fatal(err)
	}
	noise := get("latent_noise")
	latent := ttsMatrix{data: make([]float32, m.config.FlowSize*frames), channels: m.config.FlowSize, length: frames}
	if len(noise) != len(latent.data) {
		t.Fatalf("duration changed output length: %d want %d", len(latent.data), len(noise))
	}
	for c := 0; c < latent.channels; c++ {
		pos := 0
		for token, n := range durations {
			mean, scale := means.data[c*means.length+token], float32(math.Exp(float64(logs.data[c*logs.length+token])))
			for range n {
				i := c*frames + pos
				latent.data[i] = mean + noise[i]*scale*m.config.NoiseScale
				pos++
			}
		}
	}
	check("prior_latents", latent.data, 3e-4)
	latent, err = m.flow(ctx, latent)
	if err != nil {
		t.Fatal(err)
	}
	check("latents", latent.data, 3e-4)
	wave, err := ttsVocode(ctx, m.weights, m.config, latent)
	if err != nil {
		t.Fatal(err)
	}
	check("waveform", wave, 1e-3)
	zeroNoise := float32(0)
	var ownedPCM, savedPCM []float32
	for _, options := range []TTSOptions{{Seed: 19}, {Seed: 19, Speed: 1.25, NoiseScale: &zeroNoise, DurationNoiseScale: &zeroNoise}} {
		a, err := m.Synthesize(ctx, "Hallo Welt.", options)
		if err != nil {
			t.Fatal(err)
		}
		if len(a.Samples) == 0 || a.SampleRate != 16000 {
			t.Fatal("empty/wrong-rate audio")
		}
		if ownedPCM == nil {
			ownedPCM = a.Samples
			savedPCM = append([]float32(nil), a.Samples...)
		}
		b, err := m.Synthesize(ctx, "Hallo Welt.", options)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(a.Samples, b.Samples) {
			t.Fatal("same seed produced different samples")
		}
	}
	if !slices.Equal(ownedPCM, savedPCM) {
		t.Fatal("later synthesis changed previously returned PCM")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ownedPCM, savedPCM) {
		t.Fatal("closing the model changed returned PCM")
	}
}

func BenchmarkTTSSynthesize(b *testing.B) {
	path := os.Getenv("GOPHERLLM_TTS_MODEL")
	if path == "" {
		b.Skip("set GOPHERLLM_TTS_MODEL to benchmark local MMS weights")
	}
	m, err := OpenTTS(context.Background(), path)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { m.Close() })
	// Warm Accelerate and touch resident weights before reporting synthesis.
	if _, err := m.Synthesize(context.Background(), "Hallo Welt.", TTSOptions{Seed: 19}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	start := time.Now()
	var samples int
	for i := 0; i < b.N; i++ {
		a, err := m.Synthesize(context.Background(), "Hallo Welt.", TTSOptions{Seed: 19})
		if err != nil {
			b.Fatal(err)
		}
		samples += len(a.Samples)
	}
	elapsed := time.Since(start).Seconds()
	b.StopTimer()
	b.ReportMetric(float64(samples)/16000/float64(b.N), "audio_s/op")
	b.ReportMetric(float64(samples)/16000/elapsed, "realtime_x")
}
