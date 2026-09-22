package gopherllm

import (
	"encoding/binary"
	"math/rand"
	"strings"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/internal/yolotest"
)

func TestYOLOLoadsPyTorchCheckpoint(t *testing.T) {
	synth := yolotest.V11()
	fromPT, err := LoadYOLOPyTorch(synth.Torch([]string{"cat", "dog", "bird"}))
	if err != nil {
		t.Fatal(err)
	}
	if fromPT.Version != "yolo11" || strings.Join(fromPT.ClassNames, ",") != "cat,dog,bird" {
		t.Fatalf("loaded %s with names %v", fromPT.Version, fromPT.ClassNames)
	}
	fromST, err := LoadYOLOSafetensors(synth.Safetensors(yoloUltralyticsNames))
	if err != nil {
		t.Fatal(err)
	}
	input, err := PrepareYOLOImage(yoloSynthImage(), YOLOConfig{InputWidth: 64, InputHeight: 64})
	if err != nil {
		t.Fatal(err)
	}
	a, _, _, err := fromPT.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	b, _, _, err := fromST.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("output %d: .pt %v vs safetensors %v", i, a[i], b[i])
		}
	}
}

func TestUnpickleNeverExecutes(t *testing.T) {
	// The classic pickle RCE payload: os.system("touch /tmp/pwned").
	payload := append([]byte("cos\nsystem\n(U\x10touch /tmp/pwned"), "tR."...)
	v, err := unpickle(payload)
	if err != nil {
		t.Fatal(err)
	}
	obj, ok := v.(*pickleObject)
	if !ok || obj.class != (pickleGlobal{"os", "system"}) || len(obj.args) != 1 || obj.args[0] != "touch /tmp/pwned" {
		t.Fatalf("payload decoded to %#v, want an inert os.system object", v)
	}
}

func TestTorchCheckpointSurvivesCorruption(t *testing.T) {
	valid := yolotest.V8().Torch(nil)
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 400; i++ {
		data := append([]byte(nil), valid...)
		switch i % 3 {
		case 0:
			data = data[:rng.Intn(len(data))]
		default:
			for range 1 + rng.Intn(8) {
				data[rng.Intn(len(data))] = byte(rng.Intn(256))
			}
		}
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("corruption %d panicked: %v", i, r)
				}
			}()
			_, _ = LoadYOLOPyTorch(data)
		}()
	}
	z, err := openTorchZip(valid)
	if err != nil {
		t.Fatal(err)
	}
	pkl := z.entries["archive/data.pkl"]
	for range 300 {
		n := rng.Intn(len(pkl))
		if _, err := unpickle(pkl[:n]); err == nil {
			t.Fatalf("truncated pickle (%d of %d bytes) decoded without error", n, len(pkl))
		}
	}
}

func TestTorchTensorHonoursStridesAndHalfPrecision(t *testing.T) {
	// A 2x3 half tensor viewed transposed (size [3,2], stride [1,3]) at
	// offset 1 into a 7-element storage.
	var raw []byte
	for i := range 7 {
		raw = binary.LittleEndian.AppendUint16(raw, F32ToF16(float32(i)))
	}
	c := &torchCheckpoint{storages: map[string][]byte{"0": raw}}
	st := &pickleStorage{dtype: "HalfStorage", key: "0", numel: 7}
	got, err := c.tensorF32(&pickleTensor{storage: st, offset: 1, size: []int{3, 2}, stride: []int{1, 3}})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 4, 2, 5, 3, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("tensor = %v, want %v", got, want)
		}
	}
	if _, err := c.tensorF32(&pickleTensor{storage: st, size: []int{1 << 40, 1 << 40}, stride: []int{1, 1}}); err == nil {
		t.Fatal("expected an error for a shape larger than its storage")
	}
}
