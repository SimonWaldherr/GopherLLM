// Package yolotest builds small random-weight YOLO checkpoints (YOLOv8 and
// YOLO11 layouts) as safetensors or torch.save bytes, so tests of the
// loader, the HTTP routes and the demo can run without redistributing any
// real (AGPL-licensed) weights.
package yolotest

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

// Checkpoint is a random-weight checkpoint under Ultralytics' layer-indexed
// state_dict names without the "model." prefix ("0.conv.weight", ...).
type Checkpoint struct {
	rng     *rand.Rand
	Tensors map[string][]float32
	Shapes  map[string][]int
}

func New() *Checkpoint {
	return &Checkpoint{rng: rand.New(rand.NewSource(7)), Tensors: map[string][]float32{}, Shapes: map[string][]int{}}
}

func (s *Checkpoint) add(name string, shape []int, fill func() float32) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = fill()
	}
	s.Tensors[name], s.Shapes[name] = v, shape
}

func (s *Checkpoint) gauss(scale float64) func() float32 {
	return func() float32 { return float32(s.rng.NormFloat64() * scale) }
}

// Conv adds a Conv+BN block; in=1 with out>1 makes it depthwise.
func (s *Checkpoint) Conv(name string, out, in, k int) {
	s.add(name+".conv.weight", []int{out, in, k, k}, s.gauss(1/math.Sqrt(float64(in*k*k))))
	s.add(name+".bn.weight", []int{out}, func() float32 { return 0.5 + s.rng.Float32() })
	s.add(name+".bn.bias", []int{out}, s.gauss(0.1))
	s.add(name+".bn.running_mean", []int{out}, s.gauss(0.1))
	s.add(name+".bn.running_var", []int{out}, func() float32 { return 0.5 + s.rng.Float32() })
}

func (s *Checkpoint) bare(name string, out, in int) {
	s.add(name+".weight", []int{out, in, 1, 1}, s.gauss(0.3))
	s.add(name+".bias", []int{out}, s.gauss(0.3))
}

func (s *Checkpoint) bottleneck(name string, c, hidden int) {
	s.Conv(name+".cv1", hidden, c, 3)
	s.Conv(name+".cv2", c, hidden, 3)
}

// c2f adds a C2f/C3k2 with n inner modules of hidden width c; c3k selects
// C3k inner modules (two bottlenecks each) over plain bottlenecks.
func (s *Checkpoint) c2f(name string, in, out, c, n int, c3k bool, bottleneckHidden int) {
	s.Conv(name+".cv1", 2*c, in, 1)
	for i := range n {
		m := fmt.Sprintf("%s.m.%d", name, i)
		if !c3k {
			s.bottleneck(m, c, bottleneckHidden)
			continue
		}
		s.Conv(m+".cv1", c/2, c, 1)
		s.Conv(m+".cv2", c/2, c, 1)
		s.bottleneck(m+".m.0", c/2, c/2)
		s.bottleneck(m+".m.1", c/2, c/2)
		s.Conv(m+".cv3", c, c, 1)
	}
	s.Conv(name+".cv2", out, (2+n)*c, 1)
}

func (s *Checkpoint) sppf(name string, c int) {
	s.Conv(name+".cv1", c/2, c, 1)
	s.Conv(name+".cv2", c, 2*c, 1)
}

func (s *Checkpoint) head(name string, filters [3]int, box, cls int, separable bool, classes int) {
	for i, f := range filters {
		p := fmt.Sprintf("%s.cv2.%d", name, i)
		s.Conv(p+".0", box, f, 3)
		s.Conv(p+".1", box, box, 3)
		s.bare(p+".2", 64, box)
		p = fmt.Sprintf("%s.cv3.%d", name, i)
		if separable {
			s.Conv(p+".0.0", f, 1, 3)
			s.Conv(p+".0.1", cls, f, 1)
			s.Conv(p+".1.0", cls, 1, 3)
			s.Conv(p+".1.1", cls, cls, 1)
		} else {
			s.Conv(p+".0", cls, f, 3)
			s.Conv(p+".1", cls, cls, 3)
		}
		s.bare(p+".2", classes, cls)
	}
	s.add(name+".dfl.conv.weight", []int{1, 16, 1, 1}, func() float32 { return 0 })
}

// V8 is a structurally complete YOLOv8 at 64·w = 4 channels.
func V8() *Checkpoint {
	s := New()
	s.Conv("0", 4, 3, 3)
	s.Conv("1", 8, 4, 3)
	s.c2f("2", 8, 8, 4, 1, false, 4)
	s.Conv("3", 16, 8, 3)
	s.c2f("4", 16, 16, 8, 1, false, 8)
	s.Conv("5", 32, 16, 3)
	s.c2f("6", 32, 32, 16, 1, false, 16)
	s.Conv("7", 32, 32, 3)
	s.c2f("8", 32, 32, 16, 1, false, 16)
	s.sppf("9", 32)
	s.c2f("12", 64, 32, 16, 1, false, 16)
	s.c2f("15", 48, 16, 8, 1, false, 8)
	s.Conv("16", 16, 16, 3)
	s.c2f("18", 48, 32, 16, 1, false, 16)
	s.Conv("19", 32, 32, 3)
	s.c2f("21", 64, 32, 16, 1, false, 16)
	s.head("22", [3]int{16, 32, 32}, 64, 16, false, 3)
	return s
}

// V11 is a small YOLO11 with bottleneck and C3k variants of C3k2
// and a 256-channel C2PSA, so its attention runs with two heads.
func V11() *Checkpoint {
	s := New()
	s.Conv("0", 8, 3, 3)
	s.Conv("1", 16, 8, 3)
	s.c2f("2", 16, 32, 8, 1, false, 4)
	s.Conv("3", 32, 32, 3)
	s.c2f("4", 32, 32, 8, 1, false, 4)
	s.Conv("5", 64, 32, 3)
	s.c2f("6", 64, 64, 32, 1, true, 0)
	s.Conv("7", 256, 64, 3)
	s.c2f("8", 256, 256, 128, 1, true, 0)
	s.sppf("9", 256)
	s.Conv("10.cv1", 256, 256, 1)
	s.Conv("10.m.0.attn.qkv", 256, 128, 1) // 128 + 2 heads * 2*32 key channels
	s.Conv("10.m.0.attn.proj", 128, 128, 1)
	s.Conv("10.m.0.attn.pe", 128, 1, 3)
	s.Conv("10.m.0.ffn.0", 256, 128, 1)
	s.Conv("10.m.0.ffn.1", 128, 256, 1)
	s.Conv("10.cv2", 256, 256, 1)
	s.c2f("13", 320, 64, 32, 1, false, 16)
	s.c2f("16", 96, 32, 16, 1, false, 8)
	s.Conv("17", 32, 32, 3)
	s.c2f("19", 96, 64, 32, 1, false, 16)
	s.Conv("20", 64, 64, 3)
	s.c2f("22", 320, 256, 128, 1, true, 0)
	s.head("23", [3]int{32, 64, 256}, 64, 32, true, 3)
	return s
}

// Safetensors serialises the checkpoint, naming each tensor rename(name)
// (for example "model."+name for Ultralytics naming). Like an Ultralytics
// state_dict it also carries an int64 BatchNorm counter the loader must
// skip.
func (s *Checkpoint) Safetensors(rename func(string) string) []byte {
	names := make([]string, 0, len(s.Tensors))
	for name := range s.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	header := map[string]any{}
	var body []byte
	for _, name := range names {
		begin := len(body)
		for _, v := range s.Tensors[name] {
			body = binary.LittleEndian.AppendUint32(body, math.Float32bits(v))
		}
		header[rename(name)] = map[string]any{"dtype": "F32", "shape": s.Shapes[name], "data_offsets": []int{begin, len(body)}}
	}
	header[rename("0.bn.num_batches_tracked")] = map[string]any{"dtype": "I64", "shape": []int{}, "data_offsets": []int{len(body), len(body) + 8}}
	body = append(body, make([]byte, 8)...)
	h, err := json.Marshal(header)
	if err != nil {
		panic(err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	return append(append(out, h...), body...)
}

// picklew emits protocol-2 pickle opcodes, enough to reproduce the shape of
// an Ultralytics torch.save checkpoint in tests.
type picklew struct{ bytes.Buffer }

func (p *picklew) op(b ...byte) { p.Write(b) }

func (p *picklew) str(s string) {
	p.op('X')
	_ = binary.Write(&p.Buffer, binary.LittleEndian, uint32(len(s)))
	p.WriteString(s)
}

func (p *picklew) int(n int) {
	p.op('J')
	_ = binary.Write(&p.Buffer, binary.LittleEndian, int32(n))
}

func (p *picklew) global(module, name string) { p.WriteString("c" + module + "\n" + name + "\n") }

func (p *picklew) ints(v []int) {
	p.op('(')
	for _, n := range v {
		p.int(n)
	}
	p.op('t')
}

// tensor emits _rebuild_parameter(_rebuild_tensor_v2(persistent storage,
// offset, size, stride, False, OrderedDict()), True, OrderedDict()), the
// way torch.save writes an nn.Parameter.
func (p *picklew) tensor(storageType, key string, numel, offset int, size, stride []int) {
	p.global("torch._utils", "_rebuild_parameter")
	p.op('(')
	p.global("torch._utils", "_rebuild_tensor_v2")
	p.op('(')
	p.op('(')
	p.str("storage")
	p.global("torch", storageType)
	p.str(key)
	p.str("cpu")
	p.int(numel)
	p.op('t', 'Q')
	p.int(offset)
	p.ints(size)
	p.ints(stride)
	p.op(0x89)
	p.global("collections", "OrderedDict")
	p.op(')', 'R', 't', 'R') // OrderedDict(), args tuple, _rebuild_tensor_v2
	p.op(0x88)
	p.global("collections", "OrderedDict")
	p.op(')', 'R', 't', 'R') // (tensor, True, OrderedDict()), _rebuild_parameter
}

// yoloTorchModule is one nn.Module in a synthetic checkpoint tree.
type yoloTorchModule struct {
	children map[string]*yoloTorchModule
	tensors  map[string]string // parameter name -> full state_dict name
}

func (m *yoloTorchModule) child(name string) *yoloTorchModule {
	if m.children == nil {
		m.children = map[string]*yoloTorchModule{}
	}
	c, ok := m.children[name]
	if !ok {
		c = &yoloTorchModule{}
		m.children[name] = c
	}
	return c
}

// Torch writes s as an Ultralytics-style .pt: a zip holding
// archive/data.pkl, whose root dict's "model" is a DetectionModel object
// tree, and one FloatStorage file per tensor.
func (s *Checkpoint) Torch(names []string) []byte {
	root := &yoloTorchModule{}
	keys := make([]string, 0, len(s.Tensors))
	for name := range s.Tensors {
		keys = append(keys, name)
	}
	sort.Strings(keys)
	for _, name := range keys {
		parts := strings.Split("model."+name, ".")
		m := root
		for _, part := range parts[:len(parts)-1] {
			m = m.child(part)
		}
		if m.tensors == nil {
			m.tensors = map[string]string{}
		}
		m.tensors[parts[len(parts)-1]] = name
	}

	var p picklew
	storages := map[string][]byte{}
	var emit func(m *yoloTorchModule, class string)
	emit = func(m *yoloTorchModule, class string) {
		mod, cls, _ := strings.Cut(class, " ")
		p.global(mod, cls)
		p.op(')', 0x81) // NEWOBJ
		p.op('}', '(')
		p.str("training")
		p.op(0x89)
		p.str("_parameters")
		p.global("collections", "OrderedDict")
		p.op(')', 'R', '(')
		params := make([]string, 0, len(m.tensors))
		for k := range m.tensors {
			params = append(params, k)
		}
		sort.Strings(params)
		for _, k := range params {
			full := m.tensors[k]
			key := strconv.Itoa(len(storages))
			var raw []byte
			for _, v := range s.Tensors[full] {
				raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
			}
			storages[key] = raw
			shape := s.Shapes[full]
			stride := make([]int, len(shape))
			acc := 1
			for i := len(shape) - 1; i >= 0; i-- {
				stride[i], acc = acc, acc*shape[i]
			}
			p.str(k)
			p.tensor("FloatStorage", key, acc, 0, shape, stride)
		}
		p.op('u')
		p.str("_buffers")
		p.global("collections", "OrderedDict")
		p.op(')', 'R')
		if len(m.tensors) > 0 && m.tensors["running_var"] != "" {
			// A LongStorage num_batches_tracked, as every BatchNorm2d has.
			key := strconv.Itoa(len(storages))
			storages[key] = make([]byte, 8)
			p.op('(')
			p.str("num_batches_tracked")
			p.tensor("LongStorage", key, 1, 0, nil, nil)
			p.op('u')
		}
		p.str("_modules")
		p.global("collections", "OrderedDict")
		p.op(')', 'R', '(')
		kids := make([]string, 0, len(m.children))
		for k := range m.children {
			kids = append(kids, k)
		}
		sort.Strings(kids)
		for _, k := range kids {
			p.str(k)
			emit(m.children[k], "torch.nn.modules.container Module")
		}
		p.op('u')
		if class == "ultralytics.nn.tasks DetectionModel" {
			p.str("names")
			p.op('}', '(')
			for i, n := range names {
				p.int(i)
				p.str(n)
			}
			p.op('u')
		}
		p.op('u', 'b') // SETITEMS into the state dict, then BUILD
	}
	p.op(0x80, 2, '}', '(')
	p.str("version")
	p.str("8.3.0")
	p.str("model")
	emit(root, "ultralytics.nn.tasks DetectionModel")
	p.str("ema")
	p.op('N')
	p.op('u', '.')

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	write := func(name string, data []byte) {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: name, Method: zip.Store})
		if err != nil {
			panic(err)
		}
		if _, err := w.Write(data); err != nil {
			panic(err)
		}
	}
	write("archive/data.pkl", p.Bytes())
	write("archive/byteorder", []byte("little"))
	for key, raw := range storages {
		write("archive/data/"+key, raw)
	}
	if err := zw.Close(); err != nil {
		panic(err)
	}
	return buf.Bytes()
}
