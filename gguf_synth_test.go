package main

import (
	"bytes"
	"encoding/binary"
	"math"
)

// This file provides helpers for constructing in-memory GGUF files so the
// parser, config inference, weight loader, and forward pass can be exercised
// without a multi-gigabyte model on disk.

// GGUF metadata value type tags (see parseGGUF).
const (
	ggufU8   uint32 = 0
	ggufI8   uint32 = 1
	ggufU16  uint32 = 2
	ggufI16  uint32 = 3
	ggufU32  uint32 = 4
	ggufI32  uint32 = 5
	ggufF32  uint32 = 6
	ggufBool uint32 = 7
	ggufStr  uint32 = 8
	ggufArr  uint32 = 9
	ggufU64  uint32 = 10
	ggufI64  uint32 = 11
	ggufF64  uint32 = 12
)

type ggufKV struct {
	key string
	typ uint32
	val any
}

// ggufArray is the value for an array-typed (type 9) metadata entry.
type ggufArray struct {
	elem  uint32
	items []any
}

type ggufTensor struct {
	name  string
	dims  []uint64
	dtype GGMLType
	data  []byte
}

func ggufPutU16(b *bytes.Buffer, v uint16) {
	var x [2]byte
	binary.LittleEndian.PutUint16(x[:], v)
	b.Write(x[:])
}
func ggufPutU32(b *bytes.Buffer, v uint32) {
	var x [4]byte
	binary.LittleEndian.PutUint32(x[:], v)
	b.Write(x[:])
}
func ggufPutU64(b *bytes.Buffer, v uint64) {
	var x [8]byte
	binary.LittleEndian.PutUint64(x[:], v)
	b.Write(x[:])
}
func ggufPutStr(b *bytes.Buffer, s string) {
	ggufPutU64(b, uint64(len(s)))
	b.WriteString(s)
}

func ggufWriteValue(b *bytes.Buffer, typ uint32, val any) {
	switch typ {
	case ggufU8:
		b.WriteByte(val.(uint8))
	case ggufI8:
		b.WriteByte(byte(val.(int8)))
	case ggufU16:
		ggufPutU16(b, val.(uint16))
	case ggufI16:
		ggufPutU16(b, uint16(val.(int16)))
	case ggufU32:
		ggufPutU32(b, val.(uint32))
	case ggufI32:
		ggufPutU32(b, uint32(val.(int32)))
	case ggufF32:
		ggufPutU32(b, math.Float32bits(val.(float32)))
	case ggufBool:
		if val.(bool) {
			b.WriteByte(1)
		} else {
			b.WriteByte(0)
		}
	case ggufStr:
		ggufPutStr(b, val.(string))
	case ggufArr:
		a := val.(ggufArray)
		ggufPutU32(b, a.elem)
		ggufPutU64(b, uint64(len(a.items)))
		for _, it := range a.items {
			ggufWriteValue(b, a.elem, it)
		}
	case ggufU64:
		ggufPutU64(b, val.(uint64))
	case ggufI64:
		ggufPutU64(b, uint64(val.(int64)))
	case ggufF64:
		var x [8]byte
		binary.LittleEndian.PutUint64(x[:], math.Float64bits(val.(float64)))
		b.Write(x[:])
	default:
		panic("ggufWriteValue: unknown type")
	}
}

// buildGGUF assembles a valid GGUF byte stream. Tensor data is laid out
// contiguously in the order given (offsets relative to the aligned data start).
func buildGGUF(version uint32, kvs []ggufKV, tensors []ggufTensor) []byte {
	var head bytes.Buffer
	head.WriteString("GGUF")
	ggufPutU32(&head, version)
	ggufPutU64(&head, uint64(len(tensors)))
	ggufPutU64(&head, uint64(len(kvs)))
	for _, kv := range kvs {
		ggufPutStr(&head, kv.key)
		ggufPutU32(&head, kv.typ)
		ggufWriteValue(&head, kv.typ, kv.val)
	}
	var tdata bytes.Buffer
	offsets := make([]uint64, len(tensors))
	for i, t := range tensors {
		offsets[i] = uint64(tdata.Len())
		tdata.Write(t.data)
	}
	for i, t := range tensors {
		ggufPutStr(&head, t.name)
		ggufPutU32(&head, uint32(len(t.dims)))
		for _, d := range t.dims {
			ggufPutU64(&head, d)
		}
		ggufPutU32(&head, uint32(t.dtype))
		ggufPutU64(&head, offsets[i])
	}
	const align = 32
	if pad := (align - head.Len()%align) % align; pad > 0 {
		head.Write(make([]byte, pad))
	}
	head.Write(tdata.Bytes())
	return head.Bytes()
}

func f32Bytes(f []float32) []byte {
	b := make([]byte, 4*len(f))
	for i, v := range f {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// smallWeights returns n deterministic, small, non-degenerate float32 values.
func smallWeights(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(((i+seed)%13)-6) / 20
	}
	return out
}

func onesF32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

// buildTinyLlamaGGUF builds a minimal but structurally complete 1-layer llama
// model (F32 weights) that RunnerFromGGUFBytes can load and run.
func buildTinyLlamaGGUF() []byte {
	const (
		dim    = 8
		heads  = 2
		kv     = 2
		hdim   = dim / heads // 4
		hidden = 16
		vocab  = 16
	)
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	special := []string{"<unk>", "<s>", "</s>"}
	for i := 0; i < vocab; i++ {
		if i < len(special) {
			toks[i] = special[i]
		} else {
			toks[i] = string(rune('a' + (i - len(special))))
		}
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "llama"},
		{"general.name", ggufStr, "tiny"},
		{"llama.embedding_length", ggufU32, uint32(dim)},
		{"llama.block_count", ggufU32, uint32(1)},
		{"llama.attention.head_count", ggufU32, uint32(heads)},
		{"llama.attention.head_count_kv", ggufU32, uint32(kv)},
		{"llama.attention.key_length", ggufU32, uint32(hdim)},
		{"llama.attention.value_length", ggufU32, uint32(hdim)},
		{"llama.feed_forward_length", ggufU32, uint32(hidden)},
		{"llama.context_length", ggufU32, uint32(32)},
		{"llama.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{"llama.rope.freq_base", ggufF32, float32(10000)},
		{"llama.rope.dimension_count", ggufU32, uint32(hdim)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.add_bos_token", ggufBool, true},
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		// GGUF stores dims as [cols(in), rows(out)].
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim),
		f32t("output.weight", vocab, dim, 2),
		vec("blk.0.attn_norm.weight", dim),
		f32t("blk.0.attn_q.weight", heads*hdim, dim, 3),
		f32t("blk.0.attn_k.weight", kv*hdim, dim, 4),
		f32t("blk.0.attn_v.weight", kv*hdim, dim, 5),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 6),
		vec("blk.0.ffn_norm.weight", dim),
		f32t("blk.0.ffn_gate.weight", hidden, dim, 7),
		f32t("blk.0.ffn_up.weight", hidden, dim, 8),
		f32t("blk.0.ffn_down.weight", dim, hidden, 9),
	}
	return buildGGUF(3, kvs, tensors)
}
