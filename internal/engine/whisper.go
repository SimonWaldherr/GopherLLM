package engine

import (
	"context"
	"fmt"
	"math"
	"strings"
)

const whisperMaxSamples = 30 * 16000

// Language tokens use the ordering of the OpenAI Whisper vocabulary.
var whisperLanguageCodes = strings.Fields("en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue")

func (m *WhisperModel) whisperLanguage(language string) (int, error) {
	language = strings.ToLower(strings.TrimSpace(language))
	if language == "" || language == "auto" {
		return -1, nil
	}
	for i, code := range whisperLanguageCodes[:m.languages] {
		if language == code && (m.vocab != 51864 || code == "en") {
			return m.start + 1 + i, nil
		}
	}
	return 0, fmt.Errorf("Whisper language %q is unavailable for this checkpoint", language)
}

func whisperSuppressPiece(piece string) bool {
	// Keep ordinary punctuation needed for readable transcripts. A piece can
	// contain raw partial UTF-8 bytes, so suppress only exact symbol pieces.
	if piece == " -" || piece == " '" {
		return true
	}
	piece = strings.TrimPrefix(piece, " ")
	switch piece {
	case "", "#", "$", "%", "&", "*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
		"(", ")", "♪", "♫", "♬", "「", "」", "『", "』", "<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[", "('", "\"":
		return true
	}
	return false
}

func (m *WhisperModel) whisperMel(ctx context.Context, samples []float32) ([]float32, error) {
	const nfft, hop, frames = 400, 160, whisperMaxSamples / 160
	padded := make([]float32, whisperMaxSamples+nfft)
	copy(padded[nfft/2:], samples)
	// Centered STFT reflects the first frame and the right boundary of the
	// fixed-size clip, which is zero-padded before reflection.
	for i := range nfft / 2 {
		padded[i] = padded[nfft-i]
		padded[len(padded)-1-i] = padded[len(padded)-nfft+i-1]
	}
	mel := make([]float32, frames*m.mels)
	task := speechMelTask{ctx: ctx, dft: cachedSpeechDFT(nfft), samples: padded,
		window: voxtralPeriodicHannWindow(nfft), filterbank: m.filterbank, mel: mel, frames: frames, hop: hop}
	if err := task.run(); err != nil {
		return nil, err
	}
	peak := float32(-10)
	for i, value := range mel {
		mel[i] = float32(math.Log10(float64(max(value, 1e-10))))
		peak = max(peak, mel[i])
	}
	for i := range mel {
		mel[i] = (max(mel[i], peak-8) + 4) / 4
	}
	return mel, ctx.Err()
}

func (l whisperLinear) batch(x [][]float32) [][]float32 {
	out := audioMatrixRows(len(x), l.weight.Rows)
	parakeetMatvecBatch(l.weight, x, out)
	if l.bias != nil {
		for _, row := range out {
			addInPlace(row, l.bias)
		}
	}
	return out
}

func whisperConvolve(l whisperLinear, x [][]float32, stride int) [][]float32 {
	n, in := len(x), len(x[0])
	patches := audioMatrixRows((n+stride-1)/stride, 3*in)
	for i, patch := range patches {
		center := i * stride
		for tap := range 3 {
			if j := center + tap - 1; j >= 0 && j < n {
				for c, value := range x[j] {
					patch[3*c+tap] = value
				}
			}
		}
	}
	out := l.batch(patches)
	whisperGELU(out)
	return out
}

func whisperGELU(rows [][]float32) {
	parallelChunks(len(rows), func(start, end int) {
		for _, row := range rows[start:end] {
			for j, value := range row {
				row[j] = value * .5 * (1 + float32(math.Erf(float64(value)*math.Sqrt2/2)))
			}
		}
	})
}

func (n whisperNorm) apply(x [][]float32) [][]float32 {
	out := audioMatrixRows(len(x), len(n.weight))
	for i, row := range x {
		layerNormInto(row, n.weight, n.bias, 1e-5, &out[i])
	}
	return out
}

func whisperResidual(dst, src [][]float32) {
	for i := range dst {
		addInPlace(dst[i], src[i])
	}
}

// Input and output use time-major, head-interleaved rows. The caller supplies
// exactly the visible K/V prefix for single-token decoder self-attention.
func whisperAttend(ctx context.Context, query [][]float32, key, value []float32, heads int) [][]float32 {
	n, width := len(query), len(query[0])
	total, dim := len(key)/width, width/heads
	out := audioMatrixRows(n, width)
	flatQ, _ := contiguousBatchRows(query, width)
	flatOut, _ := contiguousBatchRows(out, width)
	var tmp []float32
	if imageAttentionBLAS(ctx, flatQ, key, value, flatOut, n, total, heads, dim, 0, 0, &tmp) {
		return out
	}
	scale := float32(1 / math.Sqrt(float64(dim)))
	workers := min(max(1, numThreads()), n*heads)
	parallelChunks(workers, func(start, end int) {
		scores := make([]float32, total)
		for worker := start; worker < end; worker++ {
			for job := worker; job < n*heads; job += workers {
				if ctx.Err() != nil {
					return
				}
				row, head := job/heads, job%heads
				first := head * dim
				q := query[row][first : first+dim]
				dst := out[row][first : first+dim]
				peak := float32(math.Inf(-1))
				for j := range total {
					o := j*width + first
					scores[j] = DotF32(q, key[o:o+dim]) * scale
					peak = max(peak, scores[j])
				}
				softmaxF64InPlace(scores, peak)
				for j, p := range scores {
					o := j*width + first
					AxpyF32(dst, p, value[o:o+dim])
				}
			}
		}
	})
	return out
}

func (m *WhisperModel) whisperEncode(ctx context.Context, samples []float32) ([][]float32, error) {
	mel, err := m.whisperMel(ctx, samples)
	if err != nil {
		return nil, err
	}
	frames := m.audioCtx * 2
	x := audioMatrixRows(frames, m.mels)
	for t, row := range x {
		for c := range row {
			row[c] = mel[c*frames+t]
		}
	}
	x = whisperConvolve(m.conv1, x, 1)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	x = whisperConvolve(m.conv2, x, 2)
	for i, row := range x {
		addInPlace(row, m.audioPos[i*m.audioDim:(i+1)*m.audioDim])
	}
	for _, layer := range m.encoder {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := layer.selfNorm.apply(x)
		q, k, v := layer.self.query.batch(n), layer.self.key.batch(n), layer.self.value.batch(n)
		key, _ := contiguousBatchRows(k, m.audioDim)
		value, _ := contiguousBatchRows(v, m.audioDim)
		whisperResidual(x, layer.self.out.batch(whisperAttend(ctx, q, key, value, m.audioHeads)))
		whisperResidual(x, layer.whisperMLP(x))
	}
	return m.audioNorm.apply(x), ctx.Err()
}

func (b whisperBlock) whisperMLP(x [][]float32) [][]float32 {
	hidden := b.expand.batch(b.mlpNorm.apply(x))
	whisperGELU(hidden)
	return b.contract.batch(hidden)
}

type whisperCache struct {
	key, value           []float32
	crossKey, crossValue []float32
}

func (m *WhisperModel) whisperStep(ctx context.Context, cache []whisperCache, token, pos int, logits *[]float32) error {
	dim := m.textDim
	x := audioMatrixRows(1, dim)
	m.embedding.RowInto(token, dim, &x[0])
	addInPlace(x[0], m.textPos[pos*dim:(pos+1)*dim])
	for i, layer := range m.decoder {
		if err := ctx.Err(); err != nil {
			return err
		}
		c := &cache[i]
		n := layer.selfNorm.apply(x)
		q, k, v := layer.self.query.batch(n), layer.self.key.batch(n), layer.self.value.batch(n)
		copy(c.key[pos*dim:], k[0])
		copy(c.value[pos*dim:], v[0])
		limit := (pos + 1) * dim
		whisperResidual(x, layer.self.out.batch(whisperAttend(ctx, q, c.key[:limit], c.value[:limit], m.textHeads)))
		q = layer.cross.query.batch(layer.crossNorm.apply(x))
		whisperResidual(x, layer.cross.out.batch(whisperAttend(ctx, q, c.crossKey, c.crossValue, m.textHeads)))
		whisperResidual(x, layer.whisperMLP(x))
	}
	if logits != nil {
		m.embedding.MatvecInto(m.textNorm.apply(x)[0], logits)
	}
	return ctx.Err()
}

// Transcribe runs greedy, timestamp-free transcription of one normalized
// 16 kHz mono clip. An empty language or "auto" runs language detection.
func (m *WhisperModel) Transcribe(ctx context.Context, samples []float32, language string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if m.mapping == nil {
		return "", fmt.Errorf("Whisper model is closed")
	}
	if len(samples) == 0 || len(samples) > whisperMaxSamples {
		return "", fmt.Errorf("Whisper expects 1 to %d mono 16 kHz samples", whisperMaxSamples)
	}
	lang, err := m.whisperLanguage(language)
	if err != nil {
		return "", err
	}
	silent := true
	for _, x := range samples {
		if !finite32(x) {
			return "", fmt.Errorf("Whisper requires finite PCM samples")
		}
		silent = silent && x == 0
	}
	if silent {
		return "", nil
	}
	audio, err := m.whisperEncode(ctx, samples)
	if err != nil {
		return "", err
	}
	cache := make([]whisperCache, len(m.decoder))
	for i, layer := range m.decoder {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		c := &cache[i]
		c.key, c.value = make([]float32, m.textCtx*m.textDim), make([]float32, m.textCtx*m.textDim)
		c.crossKey, _ = contiguousBatchRows(layer.cross.key.batch(audio), m.textDim)
		c.crossValue, _ = contiguousBatchRows(layer.cross.value.batch(audio), m.textDim)
	}
	transcribe := m.start + 2 + m.languages
	noSpeech, noTimestamps := transcribe+3, transcribe+4
	var logits []float32
	if err := m.whisperStep(ctx, cache, m.start, 0, &logits); err != nil {
		return "", err
	}
	noSpeechProb := math.Exp(float64(logits[noSpeech]) - whisperLogSum(logits))
	prefix := []int{noTimestamps}
	if m.vocab != 51864 {
		if lang == -1 {
			lang = m.start + 1
			for id := lang + 1; id < m.start+1+m.languages; id++ {
				if logits[id] > logits[lang] {
					lang = id
				}
			}
		}
		prefix = []int{lang, transcribe, noTimestamps}
	}
	for i, token := range prefix {
		var dst *[]float32
		if i == len(prefix)-1 {
			dst = &logits
		}
		if err := m.whisperStep(ctx, cache, token, i+1, dst); err != nil {
			return "", err
		}
	}
	var text strings.Builder
	var totalLogProb float64
	for step := range m.textCtx / 2 {
		pos := len(prefix) + 1 + step
		scores := logits[:m.end+1]
		for id, suppress := range m.suppress {
			if suppress || step == 0 && m.pieces[id] == " " {
				scores[id] = float32(math.Inf(-1))
			}
		}
		if step == 0 {
			scores[m.end] = float32(math.Inf(-1))
		}
		best := 0
		for id, score := range scores {
			if score > scores[best] {
				best = id
			}
		}
		if !finite32(scores[best]) {
			return "", fmt.Errorf("Whisper returned invalid logits")
		}
		totalLogProb += float64(scores[best]) - whisperLogSum(scores)
		if best != m.end {
			text.WriteString(m.pieces[best])
		}
		if best == m.end || step+1 == m.textCtx/2 {
			if noSpeechProb > .6 && totalLogProb/float64(step+1) < -1 {
				return "", nil
			}
			break
		}
		if err := m.whisperStep(ctx, cache, best, pos, &logits); err != nil {
			return "", err
		}
	}
	return strings.TrimSpace(strings.ToValidUTF8(text.String(), "\uFFFD")), ctx.Err()
}

func whisperLogSum(scores []float32) float64 {
	peak := float32(math.Inf(-1))
	for _, score := range scores {
		peak = max(peak, score)
	}
	var sum float64
	for _, score := range scores {
		sum += math.Exp(float64(score - peak))
	}
	return float64(peak) + math.Log(sum)
}
