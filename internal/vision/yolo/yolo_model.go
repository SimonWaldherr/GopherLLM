package yolo

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/internal/formats/safetensors"
)

// This file is a native executor for Ultralytics YOLO detection networks, so
// that PrepareYOLOImage and the head decoder in yolo.go have something to
// run in between. It needs nothing beyond a safetensors file: no ONNX
// runtime, no PyTorch. Supported generations:
//
//   - YOLOv8 (C2f backbone and neck, SPPF, DFL head)
//   - YOLO11 (C3k2 with bottleneck or C3k blocks, C2PSA spatial attention,
//     depthwise-separable class branches in the head)
//
// Tensors are addressed by Ultralytics' own state_dict names without the
// leading "model." ("0.conv.weight", "2.m.0.cv1.conv.weight", ...). Two file
// naming schemes resolve to those:
//
//   - Ultralytics' state_dict saved with safetensors.torch.save_file
//     (model.0.conv.weight, ...), for any supported generation
//   - candle's YOLOv8 export, as published in lmz/candle-yolo-v8
//     (net.b1.0.conv.weight, fpn.n1..., head.cv2.0.0...)
//
// The generation is recognised from which tensors exist; the size variant
// (n/s/m/l/x), class count and DFL register count come from tensor shapes,
// so a custom-trained checkpoint loads the same way as the COCO ones.
// BatchNorm is folded into each convolution at load time; a checkpoint that
// was already fused (conv with bias, no bn tensors) is accepted too.

// yoloBNEps is the BatchNorm epsilon Ultralytics initialises every
// BatchNorm2d with (and candle mirrors), not PyTorch's 1e-5 default.
const yoloBNEps = 1e-3

type yoloOp int

const (
	yoloOpConv yoloOp = iota
	yoloOpC2f         // C2f (v8) or C3k2 (11)
	yoloOpSPPF
	yoloOpC2PSA
	yoloOpUpsample
	yoloOpConcat
	yoloOpDetect
)

// yoloLayerSpec is one row of an Ultralytics model YAML: which module runs,
// which earlier layers feed it (-1 is the previous one), and the few
// constructor arguments that cannot be read from weight shapes.
type yoloLayerSpec struct {
	op       yoloOp
	from     []int
	stride   int  // yoloOpConv
	shortcut bool // yoloOpC2f
}

func yoloSpec(op yoloOp, from ...int) yoloLayerSpec {
	if len(from) == 0 {
		from = []int{-1}
	}
	return yoloLayerSpec{op: op, from: from, stride: 1}
}

func yoloDown() yoloLayerSpec { s := yoloSpec(yoloOpConv); s.stride = 2; return s }

func yoloC2fSpec(shortcut bool) yoloLayerSpec {
	s := yoloSpec(yoloOpC2f)
	s.shortcut = shortcut
	return s
}

// yoloV8Layers is ultralytics/cfg/models/v8/yolov8.yaml.
var yoloV8Layers = []yoloLayerSpec{
	yoloDown(), yoloDown(), yoloC2fSpec(true), yoloDown(), yoloC2fSpec(true), // 0-4
	yoloDown(), yoloC2fSpec(true), yoloDown(), yoloC2fSpec(true), yoloSpec(yoloOpSPPF), // 5-9
	yoloSpec(yoloOpUpsample), yoloSpec(yoloOpConcat, -1, 6), yoloC2fSpec(false), // 10-12
	yoloSpec(yoloOpUpsample), yoloSpec(yoloOpConcat, -1, 4), yoloC2fSpec(false), // 13-15 (P3)
	yoloDown(), yoloSpec(yoloOpConcat, -1, 12), yoloC2fSpec(false), // 16-18 (P4)
	yoloDown(), yoloSpec(yoloOpConcat, -1, 9), yoloC2fSpec(false), // 19-21 (P5)
	yoloSpec(yoloOpDetect, 15, 18, 21), // 22
}

// yolo11Layers is ultralytics/cfg/models/11/yolo11.yaml. C3k2 defaults to
// shortcut=True everywhere; the YAML's boolean argument is c3k, which this
// loader reads from the weights instead.
var yolo11Layers = []yoloLayerSpec{
	yoloDown(), yoloDown(), yoloC2fSpec(true), yoloDown(), yoloC2fSpec(true), // 0-4
	yoloDown(), yoloC2fSpec(true), yoloDown(), yoloC2fSpec(true), yoloSpec(yoloOpSPPF), // 5-9
	yoloSpec(yoloOpC2PSA),                                                      // 10
	yoloSpec(yoloOpUpsample), yoloSpec(yoloOpConcat, -1, 6), yoloC2fSpec(true), // 11-13
	yoloSpec(yoloOpUpsample), yoloSpec(yoloOpConcat, -1, 4), yoloC2fSpec(true), // 14-16 (P3)
	yoloDown(), yoloSpec(yoloOpConcat, -1, 13), yoloC2fSpec(true), // 17-19 (P4)
	yoloDown(), yoloSpec(yoloOpConcat, -1, 10), yoloC2fSpec(true), // 20-22 (P5)
	yoloSpec(yoloOpDetect, 16, 19, 22), // 23
}

// yoloLayer is one built layer; exactly the field its op needs is set.
type yoloLayer struct {
	spec  yoloLayerSpec
	conv  *yoloConv
	block yoloBlock // C2f/C3k2, SPPF, C2PSA
}

// YOLOModel is a loaded YOLO detection network. It is immutable after
// loading and safe for concurrent use: every forward pass allocates its own
// activations.
type YOLOModel struct {
	// Version is "yolov8" or "yolo11".
	Version string
	// NumClasses is the detection head's class count (80 for COCO models).
	NumClasses int
	// ClassNames, when set, labels detections whose YOLOConfig carries no
	// ClassNames of its own. A .pt checkpoint supplies its own; safetensors
	// files carry no label list, so there it stays nil (a custom model's
	// classes are not COCO's). Assign COCOClassNames for the stock
	// safetensors checkpoints.
	ClassNames []string

	layers         []yoloLayer
	regMax         int
	boxes, classes [3]yoloChain
}

// yoloStrides are the three detection levels' downsampling factors (P3-P5).
var yoloStrides = [3]int{8, 16, 32}

// LoadYOLO reads a YOLO detection checkpoint: a safetensors file, or a
// PyTorch .pt file as Ultralytics writes it (including custom-trained
// best.pt files), recognised by content rather than extension.
func LoadYOLO(path string) (*YOLOModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("loading YOLO: %w", err)
	}
	if bytes.HasPrefix(data, []byte("PK\x03\x04")) {
		return LoadYOLOPyTorch(data)
	}
	return LoadYOLOSafetensors(data)
}

// IsYOLOCheckpoint reports whether path looks like a checkpoint LoadYOLO
// accepts. It reads only a safetensors header, or a .pt file's pickle
// (which is small; the weights are separate entries), so it is cheap enough
// to run over a model directory for a catalog. A true result does not
// guarantee the load succeeds.
func IsYOLOCheckpoint(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || info.Size() < 8 {
		return false
	}
	var head [8]byte
	if _, err := io.ReadFull(f, head[:]); err != nil {
		return false
	}
	if bytes.HasPrefix(head[:], []byte("PK\x03\x04")) {
		if info.Size() > 2<<30 {
			return false
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return false
		}
		z, err := openTorchZip(data)
		if err != nil {
			return false
		}
		for name, entry := range z.entries {
			if strings.HasSuffix(name, "/data.pkl") {
				return bytes.Contains(entry, []byte("DetectionModel"))
			}
		}
		return false
	}
	n := binary.LittleEndian.Uint64(head[:])
	if n == 0 || n > 64<<20 || n > uint64(info.Size())-8 {
		return false
	}
	header := make([]byte, n)
	if _, err := io.ReadFull(f, header); err != nil {
		return false
	}
	var tensors map[string]json.RawMessage
	if json.Unmarshal(header, &tensors) != nil {
		return false
	}
	shapes := make(map[string][]int, len(tensors))
	for name := range tensors {
		shapes[name] = nil
	}
	src, err := newYOLOWeightSource(shapes, nil)
	return err == nil && (src.has("22.cv2.0.0.conv.weight") || src.has("10.m.0.attn.qkv.conv.weight"))
}

// LoadYOLOSafetensors builds a YOLO detection network from the bytes of a
// safetensors checkpoint. Weights are copied out, so data may be released
// afterwards.
func LoadYOLOSafetensors(data []byte) (*YOLOModel, error) {
	st, err := safetensors.Parse(data)
	if err != nil {
		return nil, fmt.Errorf("loading YOLO: %w", err)
	}
	shapes := make(map[string][]int, len(st.Tensors))
	for name, info := range st.Tensors {
		shapes[name] = info.Shape
	}
	src, err := newYOLOWeightSource(shapes, st.F32)
	if err != nil {
		return nil, err
	}
	return loadYOLO(src)
}

// LoadYOLOPyTorch builds a YOLO detection network from an Ultralytics .pt
// checkpoint. The pickle inside is interpreted as data only (see
// torch_checkpoint.go): no Python and no code from the file ever runs. The
// checkpoint's own class names, when present, become ClassNames.
func LoadYOLOPyTorch(data []byte) (*YOLOModel, error) {
	ckpt, err := openTorchCheckpoint(data)
	if err != nil {
		return nil, fmt.Errorf("loading YOLO: %w", err)
	}
	root, ok := ckpt.root.(*pickleDict)
	if !ok {
		return nil, fmt.Errorf("loading YOLO: checkpoint root is not a dict")
	}
	// Released and stripped checkpoints keep the network under "model";
	// mid-training ones may hold it only as the EMA copy.
	var model *pickleObject
	for _, key := range []string{"model", "ema"} {
		if v, ok := root.get(key); ok {
			if obj, ok := v.(*pickleObject); ok {
				model = obj
				break
			}
		}
	}
	if model == nil {
		return nil, fmt.Errorf("loading YOLO: checkpoint has no model or ema network")
	}
	tensors := map[string]*pickleTensor{}
	moduleStateDict(model, "", tensors)
	shapes := make(map[string][]int, len(tensors))
	for name, t := range tensors {
		shapes[name] = t.size
	}
	src, err := newYOLOWeightSource(shapes, func(name string) ([]float32, error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %s", name)
		}
		return ckpt.tensorF32(t)
	})
	if err != nil {
		return nil, err
	}
	m, err := loadYOLO(src)
	if err != nil {
		return nil, err
	}
	m.ClassNames = yoloPickleClassNames(model, m.NumClasses)
	return m, nil
}

// yoloPickleClassNames reads DetectionModel.names ({0: "person", ...} or a
// list), returning nil unless it labels every class.
func yoloPickleClassNames(model *pickleObject, classes int) []string {
	state, ok := model.state.(*pickleDict)
	if !ok {
		return nil
	}
	v, ok := state.get("names")
	if !ok {
		return nil
	}
	names := make([]string, classes)
	switch v := v.(type) {
	case *pickleDict:
		for i, k := range v.keys {
			id, ok1 := k.(int64)
			name, ok2 := v.values[i].(string)
			if ok1 && ok2 && id >= 0 && int(id) < classes {
				names[id] = name
			}
		}
	case *pickleList:
		for i, item := range v.items {
			if name, ok := item.(string); ok && i < classes {
				names[i] = name
			}
		}
	}
	for _, name := range names {
		if name == "" {
			return nil
		}
	}
	return names
}

func loadYOLO(src yoloWeightSource) (*YOLOModel, error) {
	m := &YOLOModel{}
	var specs []yoloLayerSpec
	switch {
	case src.has("10.m.0.attn.qkv.conv.weight"):
		m.Version, specs = "yolo11", yolo11Layers
	case src.has("22.cv2.0.0.conv.weight"):
		m.Version, specs = "yolov8", yoloV8Layers
	default:
		return nil, fmt.Errorf("loading YOLO: unrecognised layer layout (supported: YOLOv8 and YOLO11 detection checkpoints)")
	}
	l := yoloLoader{src: src}
	for i, spec := range specs {
		name := strconv.Itoa(i)
		layer := yoloLayer{spec: spec}
		switch spec.op {
		case yoloOpConv:
			c := l.conv(name, spec.stride, true, false)
			layer.conv = &c
		case yoloOpC2f:
			layer.block = l.c2f(name, spec.shortcut)
		case yoloOpSPPF:
			layer.block = &yoloSPPF{cv1: l.conv(name+".cv1", 1, true, false), cv2: l.conv(name+".cv2", 1, true, false), k: 5}
		case yoloOpC2PSA:
			layer.block = l.c2psa(name)
		case yoloOpDetect:
			for lvl := range 3 {
				m.boxes[lvl] = l.boxBranch(fmt.Sprintf("%s.cv2.%d", name, lvl))
				m.classes[lvl] = l.classBranch(fmt.Sprintf("%s.cv3.%d", name, lvl))
			}
		}
		m.layers = append(m.layers, layer)
	}
	if l.err != nil {
		return nil, fmt.Errorf("loading %s: %w", m.Version, l.err)
	}
	if first := m.layers[0].conv; first.in != 3 {
		return nil, fmt.Errorf("loading %s: first convolution takes %d channels, want 3 (RGB)", m.Version, first.in)
	}
	boxOut := m.boxes[0][len(m.boxes[0])-1].out
	m.NumClasses = m.classes[0][len(m.classes[0])-1].out
	if boxOut%4 != 0 || boxOut == 0 {
		return nil, fmt.Errorf("loading %s: box branch has %d outputs, want a multiple of 4", m.Version, boxOut)
	}
	m.regMax = boxOut / 4
	for lvl := 1; lvl < 3; lvl++ {
		if m.classes[lvl][len(m.classes[lvl])-1].out != m.NumClasses || m.boxes[lvl][len(m.boxes[lvl])-1].out != boxOut {
			return nil, fmt.Errorf("loading %s: detection level %d disagrees with level 0 on head shape", m.Version, lvl)
		}
	}
	return m, nil
}

// yoloWeightSource resolves Ultralytics layer-indexed tensor names against
// whichever naming scheme the checkpoint actually uses.
type yoloWeightSource struct {
	shapes map[string][]int
	load   func(name string) ([]float32, error)
	candle bool
	prefix string // Ultralytics: "model." or "model.model."
}

// yoloCandleModules maps YOLOv8 layer indices to candle's module paths
// (layers 10, 11, 13, 14, 17 and 20 are parameterless Upsample/Concat).
var yoloCandleModules = map[string]string{
	"0": "net.b1.0", "1": "net.b1.1", "2": "net.b2.0", "3": "net.b2.1",
	"4": "net.b2.2", "5": "net.b3.0", "6": "net.b3.1", "7": "net.b4.0",
	"8": "net.b4.1", "9": "net.b5.0", "12": "fpn.n1", "15": "fpn.n2",
	"16": "fpn.n3", "18": "fpn.n4", "19": "fpn.n5", "21": "fpn.n6",
	"22": "head",
}

func newYOLOWeightSource(shapes map[string][]int, load func(string) ([]float32, error)) (yoloWeightSource, error) {
	if _, ok := shapes["net.b1.0.conv.weight"]; ok {
		return yoloWeightSource{shapes: shapes, load: load, candle: true}, nil
	}
	for _, prefix := range []string{"model.", "model.model."} {
		if _, ok := shapes[prefix+"0.conv.weight"]; ok {
			return yoloWeightSource{shapes: shapes, load: load, prefix: prefix}, nil
		}
	}
	return yoloWeightSource{}, fmt.Errorf("loading YOLO: no model.0.conv.weight or net.b1.0.conv.weight tensor; not a YOLO checkpoint in Ultralytics or candle naming")
}

func (s yoloWeightSource) resolve(name string) string {
	if !s.candle {
		return s.prefix + name
	}
	index, rest, _ := strings.Cut(name, ".")
	module, ok := yoloCandleModules[index]
	if !ok {
		return name
	}
	if index != "22" { // C2f inner modules are "m.N" upstream, "bottleneck.N" in candle
		if tail, ok := strings.CutPrefix(rest, "m."); ok {
			rest = "bottleneck." + tail
		}
	}
	return module + "." + rest
}

func (s yoloWeightSource) has(name string) bool {
	_, ok := s.shapes[s.resolve(name)]
	return ok
}

func (s yoloWeightSource) tensor(name string) ([]float32, []int, error) {
	resolved := s.resolve(name)
	shape, ok := s.shapes[resolved]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %s", resolved)
	}
	v, err := s.load(resolved)
	return v, shape, err
}

// yoloLoader accumulates the first error so LoadYOLOSafetensors can read as
// a plain walk over the layer list.
type yoloLoader struct {
	src yoloWeightSource
	err error
}

// conv loads a Conv2d (name+".conv.weight", or name+".weight" for the
// head's bare output convolutions), folding a following BatchNorm
// (name+".bn.*") when the checkpoint has one. A depthwise convolution's
// weight is [channels,1,k,k].
func (l *yoloLoader) conv(name string, stride int, act, depthwise bool) yoloConv {
	if l.err != nil {
		return yoloConv{}
	}
	base := name + ".conv"
	if !l.src.has(base + ".weight") {
		base = name
	}
	w, shape, err := l.src.tensor(base + ".weight")
	if err != nil {
		l.err = err
		return yoloConv{}
	}
	if len(shape) != 4 || shape[2] != shape[3] || shape[2]%2 != 1 || (depthwise && shape[1] != 1) {
		l.err = fmt.Errorf("tensor %s has shape %v, want [out,in,k,k] with odd k (in=1 for depthwise)", l.src.resolve(base+".weight"), shape)
		return yoloConv{}
	}
	c := yoloConv{w: w, b: make([]float32, shape[0]), out: shape[0], in: shape[1], k: shape[2], stride: stride, act: act, depthwise: depthwise}
	if depthwise {
		c.in = c.out
	}
	if l.src.has(base + ".bias") {
		bias, _, err := l.src.tensor(base + ".bias")
		if err != nil || len(bias) != c.out {
			l.err = fmt.Errorf("tensor %s: bias does not match %d output channels (%v)", l.src.resolve(base+".bias"), c.out, err)
			return yoloConv{}
		}
		copy(c.b, bias)
	}
	if bn := name + ".bn"; l.src.has(bn + ".weight") {
		var params [4][]float32
		for i, field := range []string{"weight", "bias", "running_mean", "running_var"} {
			v, _, err := l.src.tensor(bn + "." + field)
			if err != nil || len(v) != c.out {
				l.err = fmt.Errorf("batchnorm %s.%s does not match %d channels (%v)", l.src.resolve(bn), field, c.out, err)
				return yoloConv{}
			}
			params[i] = v
		}
		gamma, beta, mean, variance := params[0], params[1], params[2], params[3]
		per := len(c.w) / c.out
		for o := range c.out {
			scale := gamma[o] / float32(math.Sqrt(float64(variance[o])+yoloBNEps))
			for i := range per {
				c.w[o*per+i] *= scale
			}
			c.b[o] = (c.b[o]-mean[o])*scale + beta[o]
		}
	}
	return c
}

func (l *yoloLoader) bottleneck(name string, shortcut bool) yoloBottleneck {
	b := yoloBottleneck{cv1: l.conv(name+".cv1", 1, true, false), cv2: l.conv(name+".cv2", 1, true, false)}
	b.residual = shortcut && b.cv1.in == b.cv2.out
	return b
}

// c2f loads a C2f or C3k2 block. Each inner module is a C3k when it has a
// cv3 of its own, otherwise a plain bottleneck.
func (l *yoloLoader) c2f(name string, shortcut bool) *yoloC2f {
	c := &yoloC2f{cv1: l.conv(name+".cv1", 1, true, false), cv2: l.conv(name+".cv2", 1, true, false)}
	for i := 0; l.err == nil && l.src.has(fmt.Sprintf("%s.m.%d.cv1.conv.weight", name, i)); i++ {
		inner := fmt.Sprintf("%s.m.%d", name, i)
		if !l.src.has(inner + ".cv3.conv.weight") {
			b := l.bottleneck(inner, shortcut)
			c.m = append(c.m, &b)
			continue
		}
		c3k := &yoloC3k{cv1: l.conv(inner+".cv1", 1, true, false), cv2: l.conv(inner+".cv2", 1, true, false), cv3: l.conv(inner+".cv3", 1, true, false)}
		for j := 0; l.err == nil && l.src.has(fmt.Sprintf("%s.m.%d.cv1.conv.weight", inner, j)); j++ {
			c3k.m = append(c3k.m, l.bottleneck(fmt.Sprintf("%s.m.%d", inner, j), shortcut))
		}
		c.m = append(c.m, c3k)
	}
	if l.err == nil && len(c.m) == 0 {
		l.err = fmt.Errorf("C2f block %s has no inner modules", l.src.resolve(name))
	}
	return c
}

// c2psa loads YOLO11's C2PSA. Head geometry follows Ultralytics:
// num_heads = c/64 and key_dim = head_dim*attn_ratio, the latter recovered
// here from the qkv projection's width rather than assumed.
func (l *yoloLoader) c2psa(name string) *yoloC2PSA {
	c := &yoloC2PSA{cv1: l.conv(name+".cv1", 1, true, false), cv2: l.conv(name+".cv2", 1, true, false)}
	dim := c.cv1.out / 2
	heads := max(1, dim/64)
	for i := 0; l.err == nil && l.src.has(fmt.Sprintf("%s.m.%d.attn.qkv.conv.weight", name, i)); i++ {
		b := fmt.Sprintf("%s.m.%d", name, i)
		attn := yoloAttention{
			qkv: l.conv(b+".attn.qkv", 1, false, false), proj: l.conv(b+".attn.proj", 1, false, false),
			pe: l.conv(b+".attn.pe", 1, false, true), heads: heads, headDim: dim / heads,
		}
		attn.keyDim = (attn.qkv.out - dim) / (2 * heads)
		if l.err == nil && (dim%heads != 0 || attn.keyDim <= 0 || attn.qkv.out != dim+2*heads*attn.keyDim || attn.pe.out != dim) {
			l.err = fmt.Errorf("attention %s: qkv width %d does not split into %d heads over %d channels", l.src.resolve(b+".attn"), attn.qkv.out, heads, dim)
		}
		c.m = append(c.m, yoloPSABlock{attn: attn, ffn: yoloChain{l.conv(b+".ffn.0", 1, true, false), l.conv(b+".ffn.1", 1, false, false)}})
	}
	if l.err == nil && len(c.m) == 0 {
		l.err = fmt.Errorf("C2PSA block %s has no PSA blocks", l.src.resolve(name))
	}
	return c
}

// boxBranch loads a Detect cv2 branch: two 3x3 ConvBlocks and a bare 1x1
// Conv2d producing the 4*regMax box distribution.
func (l *yoloLoader) boxBranch(name string) yoloChain {
	return yoloChain{l.conv(name+".0", 1, true, false), l.conv(name+".1", 1, true, false), l.conv(name+".2", 1, false, false)}
}

// classBranch loads a Detect cv3 branch: YOLOv8's two 3x3 ConvBlocks, or
// YOLO11's two depthwise-separable pairs (DWConv 3x3 then Conv 1x1), each
// followed by the bare 1x1 class-logit Conv2d.
func (l *yoloLoader) classBranch(name string) yoloChain {
	if l.src.has(name + ".0.0.conv.weight") {
		return yoloChain{
			l.conv(name+".0.0", 1, true, true), l.conv(name+".0.1", 1, true, false),
			l.conv(name+".1.0", 1, true, true), l.conv(name+".1.1", 1, true, false),
			l.conv(name+".2", 1, false, false),
		}
	}
	return yoloChain{l.conv(name+".0", 1, true, false), l.conv(name+".1", 1, true, false), l.conv(name+".2", 1, false, false)}
}

// Forward runs the network over a prepared image and returns the decoded
// head output as a [4+NumClasses][anchors] row-major matrix: box centre x,
// centre y, width and height in input-image pixels, then per-class sigmoid
// scores. That is exactly the layout DecodeYOLOv8 consumes. The prepared
// image's dimensions must be multiples of 32.
func (m *YOLOModel) Forward(input YOLOImage) (output []float32, rows, cols int, err error) {
	if input.Width <= 0 || input.Height <= 0 || input.Width%32 != 0 || input.Height%32 != 0 {
		return nil, 0, 0, fmt.Errorf("%s forward: input %dx%d is not a positive multiple of 32", m.Version, input.Width, input.Height)
	}
	if len(input.Pixels) != 3*input.Width*input.Height {
		return nil, 0, 0, fmt.Errorf("%s forward: %d pixels for a %dx%d RGB input", m.Version, len(input.Pixels), input.Width, input.Height)
	}
	levels, err := m.features(yoloTensor{c: 3, h: input.Height, w: input.Width, data: input.Pixels})
	if err != nil {
		return nil, 0, 0, fmt.Errorf("%s forward: %w", m.Version, err)
	}
	anchors := 0
	for _, l := range levels {
		anchors += l.plane()
	}
	rows, cols = 4+m.NumClasses, anchors
	output = make([]float32, rows*cols)
	offset := 0
	for i, x := range levels {
		box, err := m.boxes[i].forward(x)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("%s forward: box head %d: %w", m.Version, i, err)
		}
		cls, err := m.classes[i].forward(x)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("%s forward: class head %d: %w", m.Version, i, err)
		}
		m.decodeLevel(box, cls, yoloStrides[i], output, offset, cols)
		offset += x.plane()
	}
	return output, rows, cols, nil
}

// features runs every layer before Detect and returns the three feature
// maps Detect reads. A layer's output is dropped as soon as nothing later
// refers to it, which keeps peak memory near the largest few activations
// instead of the sum of all of them.
func (m *YOLOModel) features(x yoloTensor) ([3]yoloTensor, error) {
	var levels [3]yoloTensor
	lastUse := make([]int, len(m.layers))
	for i, layer := range m.layers {
		for _, f := range layer.spec.from {
			if f < 0 {
				f += i
			}
			if f >= 0 {
				lastUse[f] = max(lastUse[f], i)
			}
		}
	}
	outputs := make([]yoloTensor, len(m.layers))
	for i, layer := range m.layers {
		inputs := make([]yoloTensor, len(layer.spec.from))
		for j, f := range layer.spec.from {
			if f < 0 {
				f += i
			}
			if i == 0 {
				inputs[j] = x
			} else {
				inputs[j] = outputs[f]
			}
		}
		var out yoloTensor
		var err error
		switch layer.spec.op {
		case yoloOpConv:
			out, err = layer.conv.forward(inputs[0])
		case yoloOpC2f, yoloOpSPPF, yoloOpC2PSA:
			out, err = layer.block.forward(inputs[0])
		case yoloOpUpsample:
			out = yoloUpsample2x(inputs[0])
		case yoloOpConcat:
			out, err = yoloConcat(inputs...)
		case yoloOpDetect:
			copy(levels[:], inputs)
			return levels, nil
		}
		if err != nil {
			return levels, fmt.Errorf("layer %d: %w", i, err)
		}
		outputs[i] = out
		for f := range i {
			if lastUse[f] == i {
				outputs[f] = yoloTensor{}
			}
		}
	}
	return levels, fmt.Errorf("model has no Detect layer")
}

// decodeLevel turns one level's raw head maps into [4+nc][anchors] rows:
// the DFL expectation over regMax bins per box side gives left/top/right/
// bottom distances from the anchor centre in grid cells, which scale by the
// level's stride into input pixels.
func (m *YOLOModel) decodeLevel(box, cls yoloTensor, stride int, out []float32, offset, anchors int) {
	plane := box.plane()
	parallelChunks(box.h, func(start, end int) {
		bins := make([]float32, m.regMax)
		for gy := start; gy < end; gy++ {
			for gx := range box.w {
				p := gy*box.w + gx
				var dist [4]float32
				for side := range 4 {
					largest := float32(math.Inf(-1))
					for j := range m.regMax {
						bins[j] = box.data[(side*m.regMax+j)*plane+p]
						largest = max(largest, bins[j])
					}
					var sum, expect float32
					for j, v := range bins {
						e := yoloFastExp(v - largest)
						sum += e
						expect += e * float32(j)
					}
					dist[side] = expect / sum
				}
				ax, ay, s := float32(gx)+0.5, float32(gy)+0.5, float32(stride)
				x1, y1, x2, y2 := ax-dist[0], ay-dist[1], ax+dist[2], ay+dist[3]
				a := offset + p
				out[a] = (x1 + x2) / 2 * s
				out[anchors+a] = (y1 + y2) / 2 * s
				out[2*anchors+a] = (x2 - x1) * s
				out[3*anchors+a] = (y2 - y1) * s
				for c := range m.NumClasses {
					out[(4+c)*anchors+a] = yoloFastSigmoid(cls.data[c*plane+p])
				}
			}
		}
	})
}

// Detect letterboxes img to cfg's input size, runs the network and returns
// non-max-suppressed detections in img's pixel coordinates. Zero-valued
// cfg fields fall back to DefaultYOLOConfig's; when cfg has no ClassNames,
// m.ClassNames labels the results.
func (m *YOLOModel) Detect(img image.Image, cfg YOLOConfig) ([]Detection, error) {
	def := DefaultYOLOConfig()
	if cfg.InputWidth <= 0 {
		cfg.InputWidth = def.InputWidth
	}
	if cfg.InputHeight <= 0 {
		cfg.InputHeight = def.InputHeight
	}
	if cfg.ClassNames == nil {
		cfg.ClassNames = m.ClassNames
	}
	input, err := PrepareYOLOImage(img, cfg)
	if err != nil {
		return nil, err
	}
	out, rows, cols, err := m.Forward(input)
	if err != nil {
		return nil, err
	}
	return decodeYOLOv8(out, rows, cols, false, input, cfg)
}

// COCOClassNames are the 80 COCO detection labels, in the class-index order
// of the stock Ultralytics checkpoints.
var COCOClassNames = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
	"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
	"dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// yoloStockRepos pins the stock detection checkpoints to fixed commits:
// YOLO11 from Ultralytics' own Hugging Face repository (.pt, read by the
// data-only unpickler), YOLOv8 from lmz/candle-yolo-v8, which republishes
// Ultralytics' COCO weights as safetensors. GopherLLM does not ship any of
// them (they are AGPL-3.0); a caller downloads them on demand, and the
// pinned revisions keep a short name from silently resolving to different
// bytes later.
var yoloStockRepos = []struct{ prefix, repo, ext string }{
	{"yolo11", "Ultralytics/YOLO11@8b8ac7d1fae7468f85dbf89670dd66f41f485aab", ".pt"},
	{"yolov8", "lmz/candle-yolo-v8@be388c6fab95ae3035a039070e1b883b9c5a1325", ".safetensors"},
}

// YOLOStockModels lists the names YOLOCheckpointReference resolves.
var YOLOStockModels = []string{
	"yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
	"yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
}

// YOLOCheckpointReference maps a stock model name (see YOLOStockModels) to
// a pinned Hugging Face reference of the form "hf:owner/repo:file@revision".
// It reports false for any other name. It only builds the reference;
// fetching it is up to the caller (see the huggingface package), so this
// package stays offline.
func YOLOCheckpointReference(name string) (string, bool) {
	name = strings.ToLower(strings.TrimSpace(name))
	if !slices.Contains(YOLOStockModels, name) {
		return "", false
	}
	for _, r := range yoloStockRepos {
		if strings.HasPrefix(name, r.prefix) {
			repo, rev, _ := strings.Cut(r.repo, "@")
			return "hf:" + repo + ":" + name + r.ext + "@" + rev, true
		}
	}
	return "", false
}
