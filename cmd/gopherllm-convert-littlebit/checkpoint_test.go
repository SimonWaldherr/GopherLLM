package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/internal/engine"
	"github.com/SimonWaldherr/GopherLLM/internal/formats/gguf"
)

func fixtureFloats(shape []int, values ...float32) hfTensor {
	return hfTensor{"F32", shape, fixtureBytes(values, math.Float32bits)}
}

func fixtureBytes[T any, B ~uint32 | ~uint64](values []T, bits func(T) B) []byte {
	var data []byte
	for _, value := range values {
		data, _ = binary.Append(data, binary.LittleEndian, bits(value))
	}
	return data
}

func fixtureI32(shape []int, values ...uint32) hfTensor {
	return hfTensor{"I32", shape, fixtureBytes(values, func(v uint32) uint32 { return v })}
}

func fixtureI64(shape []int, values ...uint64) hfTensor {
	return hfTensor{"I64", shape, fixtureBytes(values, func(v uint64) uint64 { return v })}
}

func fixtureOnes(shape ...int) hfTensor {
	count := 1
	for _, d := range shape {
		count *= d
	}
	values := make([]float32, count)
	for i := range values {
		values[i] = 1
	}
	return fixtureFloats(shape, values...)
}

func fixtureCheckpoint(t *testing.T) (string, tensorIndex) {
	t.Helper()
	const layer = "model.layers.0."
	const q = layer + "self_attn.q_proj."
	dir := t.TempDir()
	writeFixtureFile(t, dir, "config.json", []byte(`{
	  "model_type":"llama", "hidden_size":4, "intermediate_size":8,
	  "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
	  "vocab_size":5, "max_position_embeddings":32, "model_dtype":"bfloat16",
	  "bos_token_id":1, "eos_token_id":2, "tie_word_embeddings":true
	}`))
	writeFixtureFile(t, dir, "littlebit_config.json", []byte(`{"quant_mod":"LittleBitLinear","quant_func":"SmoothSign","use_itq":true}`))
	writeFixtureFile(t, dir, "tokenizer.json", []byte(`{
	  "model": {"type":"BPE","vocab":{"<unk>":0,"<s>":1,"</s>":2,"a":3,"▁":4},"merges":[],"byte_fallback":true},
	  "added_tokens":[{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}]
	}`))
	writeFixtureFile(t, dir, "tokenizer_config.json", []byte(`{"add_bos_token":true,"chat_template":"{{bos_token}}"}`))
	index := tensorIndex{
		"model.embed_tokens.weight":               fixtureOnes(5, 4),
		"model.norm.weight":                       fixtureOnes(4),
		layer + "input_layernorm.weight":          fixtureOnes(4),
		layer + "post_attention_layernorm.weight": fixtureOnes(4),
		layer + "self_attn.k_proj.weight":         fixtureFloats([]int{4, 4}, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43),
		layer + "self_attn.v_proj.weight":         fixtureOnes(4, 4),
		layer + "self_attn.o_proj.weight":         fixtureOnes(4, 4),
		layer + "mlp.gate_proj.weight":            fixtureOnes(8, 4),
		layer + "mlp.up_proj.weight":              fixtureOnes(8, 4),
		layer + "mlp.down_proj.weight":            fixtureOnes(4, 8),
		q + "U_packed":                            fixtureI32([]int{4, 1}, 1, 2, 4, 0),
		q + "U_shape":                             fixtureI64([]int{2}, 4, 3),
		q + "V_packed":                            fixtureI32([]int{3, 1}, 1, 2, 4),
		q + "V_shape":                             fixtureI64([]int{2}, 3, 4),
		q + "u1":                                  fixtureFloats([]int{1, 4}, 1, 2, 3, 4),
		q + "u2":                                  fixtureOnes(1, 3),
		q + "v1":                                  fixtureOnes(1, 3),
		q + "v2":                                  {"BF16", []int{1, 4}, []byte{0x80, 0x3f, 0, 0x40, 0, 0x40, 0x80, 0x3f}},
		q + "U_R":                                 fixtureFloats([]int{4, 1}, -1, -0.5, 0, 1),
		q + "V_R":                                 {"I8", []int{1, 4}, []byte{1, 255, 1, 255}},
		q + "u1_R":                                fixtureOnes(1, 4),
		q + "u2_R":                                fixtureOnes(1, 1),
		q + "v1_R":                                fixtureOnes(1, 1),
		q + "v2_R":                                fixtureOnes(1, 4),
		q + "bias":                                fixtureFloats([]int{4}, 1, 2, 3, 4),
		q + "_split_dim_final":                    fixtureI64(nil, 3),
		q + "_eff_bit_actual":                     fixtureFloats(nil, 4),
		q + "_eff_bit_target":                     fixtureFloats(nil, 0.1),
	}
	return dir, index
}

func writeFixtureFile(t *testing.T, dir, name string, data []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeSafetensors(t *testing.T, dir, name string, tensors tensorIndex) {
	t.Helper()
	meta := make(map[string]any, len(tensors))
	var payload []byte
	for key, tensor := range tensors {
		start := len(payload)
		payload = append(payload, tensor.raw...)
		shape := tensor.shape
		if len(shape) == 0 {
			shape = []int{}
		}
		meta[key] = map[string]any{"dtype": tensor.dtype, "shape": shape, "data_offsets": []int{start, len(payload)}}
	}
	header, err := json.Marshal(meta)
	if err != nil {
		t.Fatal(err)
	}
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	file = append(append(file, header...), payload...)
	writeFixtureFile(t, dir, name, file)
}

func TestConvertCheckpointToNativeInference(t *testing.T) {
	dir, index := fixtureCheckpoint(t)
	a, b := tensorIndex{}, tensorIndex{}
	shards := make(map[string]string)
	for name, tensor := range index {
		dst, file := a, "model-00001-of-00002.safetensors"
		if strings.HasSuffix(name, "_shape") || strings.HasSuffix(name, "_R") {
			dst, file = b, "model-00002-of-00002.safetensors"
		}
		dst[name], shards[name] = tensor, file
	}
	rawIndex, err := json.Marshal(map[string]any{"weight_map": shards})
	if err != nil {
		t.Fatal(err)
	}
	writeFixtureFile(t, dir, "model.safetensors.index.json", rawIndex)
	writeSafetensors(t, dir, "model-00001-of-00002.safetensors", a)
	writeSafetensors(t, dir, "model-00002-of-00002.safetensors", b)
	path := filepath.Join(dir, "converted.gguf")
	if err := convert(dir, path, 0); err != nil {
		t.Fatal(err)
	}
	runner, _, err := engine.RunnerFromPath(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = runner.Close() })
	model := runner.GGUF()
	if v := model.GetU32("gopherllm.littlebit.variant", 0); v != 2 {
		t.Fatalf("variant = %d, want 2", v)
	}
	if v := model.Metadata["gopherllm.littlebit.original_parameters"].Value; v != uint64(16) {
		t.Fatalf("original parameters = %v, want 16", v)
	}
	if got := runner.Tokenizer().Encode("a"); !slices.Equal(got, []uint32{1, 4, 3}) {
		t.Fatalf("tokenized prompt = %v", got)
	}
	opts := engine.DefaultGenerationOptions()
	opts.MaxTokens = 2
	opts.Sampler.Temperature = 0
	out, err := runner.Generate("a", opts)
	if err != nil || out.Stats.GeneratedTokens < 1 || out.Stats.GeneratedTokens > 2 {
		t.Fatalf("generate: %+v, %v", out, err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	vectors := map[string]struct {
		dtype gguf.GGMLType
		raw   []byte
	}{
		"blk.0.attn_q.weight.lb.u":          {gguf.GGMLTypeI32, fixtureI32(nil, 1, 4, 2, 0).raw},
		"blk.0.attn_q.weight.lb.residual.u": {gguf.GGMLTypeI32, fixtureI32(nil, 1, 0, 1, 0).raw},
		"blk.0.attn_q.weight.lb.residual.v": {gguf.GGMLTypeI32, fixtureI32(nil, 10).raw},
		"blk.0.attn_q.weight.lb.v2":         {gguf.GGMLTypeBF16, index["model.layers.0.self_attn.q_proj.v2"].raw},
		"blk.0.attn_q.weight.lb.u1":         {gguf.GGMLTypeF32, fixtureFloats(nil, 1, 3, 2, 4).raw},
		"blk.0.attn_q.weight.lb.bias":       {gguf.GGMLTypeF32, fixtureFloats(nil, 1, 3, 2, 4).raw},
		"blk.0.attn_k.weight":               {gguf.GGMLTypeF32, fixtureFloats(nil, 10, 11, 12, 13, 30, 31, 32, 33, 20, 21, 22, 23, 40, 41, 42, 43).raw},
	}
	for _, info := range model.Tensors {
		want, check := vectors[info.Name]
		if !check {
			continue
		}
		delete(vectors, info.Name)
		start := model.DataOffset + int(info.Offset)
		if got := data[start : start+len(want.raw)]; info.DType != want.dtype || !bytes.Equal(got, want.raw) {
			t.Errorf("%s: got %s %x, want %s %x", info.Name, info.DType, got, want.dtype, want.raw)
		}
	}
	if len(vectors) != 0 {
		t.Fatalf("missing tensors %v", vectors)
	}
}

func TestConvertRejectsCheckpointDamage(t *testing.T) {
	const prefix = "model.layers.0.self_attn.q_proj."
	for _, tc := range []struct {
		name, want string
		change     func(tensorIndex)
	}{
		{"missing scale", "missing checkpoint tensor", func(i tensorIndex) { delete(i, prefix+"u2_R") }},
		{"padding", "nonzero padding", func(i tensorIndex) { i[prefix+"U_packed"].raw[3] = 0x80 }},
		{"shape", "logical shape", func(i tensorIndex) { i[prefix+"V_shape"] = fixtureI64([]int{2}, 3, 5) }},
		{"unconsumed tensor", "unrecognized or unused checkpoint tensor", func(i tensorIndex) { i["adapter.weight"] = fixtureOnes(1) }},
		{"NaN factor", "non-finite value", func(i tensorIndex) { i[prefix+"U_R"] = fixtureFloats([]int{4, 1}, -1, 0, 1, float32(math.NaN())) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir, index := fixtureCheckpoint(t)
			tc.change(index)
			writeSafetensors(t, dir, "model.safetensors", index)
			out := filepath.Join(dir, "converted.gguf")
			err := convert(dir, out, 1)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want %q", err, tc.want)
			}
			if _, err := os.Stat(out); !os.IsNotExist(err) {
				t.Fatal(fmt.Errorf("failed conversion left a file: %w", err))
			}
		})
	}
}

func TestConvertQwen3CheckpointTokenizerAndProjections(t *testing.T) {
	dir, index := fixtureCheckpoint(t)
	writeFixtureFile(t, dir, "config.json", []byte(`{
	  "model_type":"qwen3", "hidden_size":4, "head_dim":4, "intermediate_size":8,
	  "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
	  "vocab_size":5, "max_position_embeddings":32, "rope_theta":1000000,
	  "bos_token_id":3, "eos_token_id":3, "tie_word_embeddings":true
	}`))
	writeFixtureFile(t, dir, "tokenizer_config.json", []byte(`{"add_bos_token":false,"eos_token":"<|im_end|>","chat_template":"<|im_start|>{{message.content}}<|im_end|>"}`))
	writeFixtureFile(t, dir, "generation_config.json", []byte(`{"eos_token_id":[3,1],"pad_token_id":1}`))
	writeFixtureFile(t, dir, "tokenizer.json", []byte(`{
	  "model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},
	  "decoder":{"type":"ByteLevel"},
	  "normalizer":{"type":"NFC"},
	  "pre_tokenizer":{"type":"Sequence","pretokenizers":[
		{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
		{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}
	  ]},
	  "added_tokens":[{"id":3,"content":"<|im_end|>","special":true}]
	}`))
	index["model.layers.0.self_attn.q_norm.weight"] = fixtureOnes(4)
	index["model.layers.0.self_attn.k_norm.weight"] = fixtureOnes(4)
	writeSafetensors(t, dir, "model.safetensors", index)
	path := filepath.Join(dir, "converted.gguf")
	if err := convert(dir, path, 1); err != nil {
		t.Fatal(err)
	}
	runner, _, err := engine.RunnerFromPath(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = runner.Close() })
	tok := runner.Tokenizer()
	if runner.Architecture() != "qwen3" || tok.Pre != "qwen3" || !tok.NormalizeNFC || !slices.Equal(tok.EOGIDs, []uint32{3, 1}) {
		t.Fatalf("unexpected Qwen3 configuration: %s %+v", runner.Architecture(), tok)
	}
	if got := tok.Encode("ab"); !slices.Equal(got, []uint32{2}) {
		t.Fatalf("Qwen3 tokenized prompt = %v", got)
	}
	if tok.Vocab[4] != "[PAD4]" || tok.EOSID != 3 {
		t.Fatalf("Qwen3 did not populate special and padded token IDs: %q", tok.Vocab)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	model := runner.GGUF()
	qNames := []string{"blk.0.attn_q.weight.lb.u", "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"}
	for _, info := range model.Tensors {
		if info.Name == "blk.0.attn_q.weight.lb.u" {
			start := model.DataOffset + int(info.Offset)
			if !bytes.Equal(data[start:start+16], index["model.layers.0.self_attn.q_proj.U_packed"].raw) {
				t.Error("Qwen3 Q factor rows were permuted")
			}
		}
		qNames = slices.DeleteFunc(qNames, func(name string) bool { return name == info.Name })
	}
	if len(qNames) > 0 {
		t.Fatalf("missing tensors: %v", qNames)
	}
}

func TestPackRowMatchesGGUFGoldenWords(t *testing.T) {
	src := bytes.Repeat([]byte{1}, 33)
	src[0], src[2], src[31] = 255, 255, 255
	packed := make([]byte, 8)
	if err := packRow(packed, src, "I8"); err != nil {
		t.Fatal(err)
	}
	if want := []byte{0x05, 0, 0, 0x80, 0, 0, 0, 0}; !bytes.Equal(packed, want) {
		t.Fatalf("packed bits % x, want % x", packed, want)
	}
	clear(packed)
	floatSigns := fixtureFloats(nil, math.Float32frombits(1<<31), 0, -0.25, 0.5)
	if err := packRow(packed, floatSigns.raw, "F32"); err != nil {
		t.Fatal(err)
	}
	if packed[0] != 4 {
		t.Fatalf("signed zero must map to +1, got packed value %08b", packed[0])
	}
}
