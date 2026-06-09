package main

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

var ErrGenerationCanceled = errors.New("generation canceled")

type EmbeddingResult struct {
	Embedding  []float32
	TokenCount int
}

type ChatRole int

const (
	ChatRoleSystem ChatRole = iota
	ChatRoleUser
	ChatRoleAssistant
)

type ChatMessage struct {
	Role    ChatRole
	Content string
}

func UserMessage(content string) ChatMessage {
	return ChatMessage{Role: ChatRoleUser, Content: content}
}
func AssistantMessage(content string) ChatMessage {
	return ChatMessage{Role: ChatRoleAssistant, Content: content}
}

type GenerationOptions struct {
	MaxTokens     int
	Sampler       SamplerConfig
	Seed          uint64
	SystemPrompt  string
	StopSequences []string
}

func DefaultGenerationOptions() GenerationOptions {
	return GenerationOptions{MaxTokens: 256, Sampler: DefaultSamplerConfig(), SystemPrompt: "You are a helpful assistant."}
}

func (o GenerationOptions) Validate() error {
	if o.MaxTokens <= 0 {
		return fmt.Errorf("max_tokens must be greater than 0")
	}
	if !finite32(o.Sampler.Temperature) || o.Sampler.Temperature < 0 {
		return fmt.Errorf("temperature must be a finite number >= 0")
	}
	if !finite32(o.Sampler.TopP) || o.Sampler.TopP <= 0 || o.Sampler.TopP > 1 {
		return fmt.Errorf("top_p must be in the range (0, 1]")
	}
	if o.Sampler.TopK < 0 {
		return fmt.Errorf("top_k must be greater than or equal to 0")
	}
	if !finite32(o.Sampler.RepeatPenalty) || o.Sampler.RepeatPenalty <= 0 {
		return fmt.Errorf("repeat_penalty must be a finite number > 0")
	}
	return nil
}

func finite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

type GenerationStats struct {
	PromptTokens    int
	GeneratedTokens int
	PrefillTime     time.Duration
	DecodeTime      time.Duration
	TotalTime       time.Duration
}

type GenerationResult struct {
	Text  string
	Stats GenerationStats
}

type LoadInfo struct {
	FileSizeBytes int
	LoadTime      time.Duration
}

type loadedKind int

const (
	loadedStandard loadedKind = iota
	loadedGptOss
	loadedGemma4
)

type Runner struct {
	gguf       *GGUFFile
	arch       string
	tok        *Tokenizer
	config     Config
	kind       loadedKind
	standard   ModelWeights
	gptOss     GptOssWeights
	gemma4     Gemma4Weights
	genLock    sync.Mutex
	mappedFile *MmapFile
}

func ArchitectureSupported(arch string) bool {
	switch arch {
	case "llama", "llama2", "llama3", "mistral", "mistral3", "qwen2":
		return true
	default:
		return false
	}
}

func RunnerFromGGUFBytes(data []byte) (*Runner, error) {
	return runnerFromGGUFBytes(data, false)
}

func runnerFromGGUFBytes(data []byte, borrowQuantized bool) (*Runner, error) {
	gguf, err := ParseGGUF(data)
	if err != nil {
		return nil, err
	}
	arch, ok := gguf.GetString("general.architecture")
	if !ok || arch == "" {
		arch = "llama"
	}
	if !ArchitectureSupported(arch) {
		return nil, fmt.Errorf("unsupported architecture: %s", arch)
	}
	tok, err := TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		return nil, err
	}
	r := &Runner{gguf: gguf, arch: arch, tok: tok}
	switch arch {
	case "gpt-oss":
		config, weights, err := LoadGptOssModel(data, gguf, borrowQuantized)
		if err != nil {
			return nil, err
		}
		r.config, r.gptOss, r.kind = config, weights, loadedGptOss
	case "gemma", "gemma2", "gemma4":
		config, weights, err := LoadGemma4Model(data, gguf, borrowQuantized)
		if err != nil {
			return nil, err
		}
		r.config, r.gemma4, r.kind = config, weights, loadedGemma4
	default:
		config, weights, err := LoadModel(data, gguf, borrowQuantized)
		if err != nil {
			return nil, err
		}
		r.config, r.standard, r.kind = config, weights, loadedStandard
	}
	return r, nil
}

func RunnerFromPath(path string) (*Runner, LoadInfo, error) {
	t0 := time.Now()
	mmap, err := OpenMmap(path)
	if err != nil {
		return nil, LoadInfo{}, fmt.Errorf("failed to open model: %w", err)
	}
	r, err := runnerFromGGUFBytes(mmap.Bytes(), true)
	if err != nil {
		_ = mmap.Close()
		return nil, LoadInfo{}, err
	}
	r.mappedFile = mmap
	return r, LoadInfo{FileSizeBytes: mmap.Len(), LoadTime: time.Since(t0)}, nil
}

func (r *Runner) Architecture() string      { return r.arch }
func (r *Runner) Tokenizer() *Tokenizer     { return r.tok }
func (r *Runner) GGUF() *GGUFFile           { return r.gguf }
func (r *Runner) Config() Config            { return r.config }
func (r *Runner) ModelName() (string, bool) { return r.gguf.GetString("general.name") }

func (r *Runner) Close() error {
	if r == nil || r.mappedFile == nil {
		return nil
	}
	err := r.mappedFile.Close()
	r.mappedFile = nil
	return err
}

func (r *Runner) Generate(prompt string, options GenerationOptions) (GenerationResult, error) {
	return r.GenerateChat([]ChatMessage{UserMessage(prompt)}, options)
}

func (r *Runner) GenerateChat(messages []ChatMessage, options GenerationOptions) (GenerationResult, error) {
	return r.GenerateChatStream(messages, options, func(string) {})
}

func (r *Runner) GenerateStream(prompt string, options GenerationOptions, onToken func(string)) (GenerationResult, error) {
	return r.GenerateChatStream([]ChatMessage{UserMessage(prompt)}, options, onToken)
}

func (r *Runner) GenerateChatStream(messages []ChatMessage, options GenerationOptions, onToken func(string)) (GenerationResult, error) {
	return r.GenerateChatStreamUntil(messages, options, func(text string) bool {
		onToken(text)
		return true
	})
}

func (r *Runner) GenerateChatStreamUntil(messages []ChatMessage, options GenerationOptions, onToken func(string) bool) (GenerationResult, error) {
	r.genLock.Lock()
	defer r.genLock.Unlock()
	if err := options.Validate(); err != nil {
		return GenerationResult{}, err
	}
	if len(messages) == 0 {
		return GenerationResult{}, fmt.Errorf("no prompt provided")
	}
	totalStart := time.Now()
	tokens := r.renderMessages(messages, options.SystemPrompt)
	if len(tokens) == 0 {
		return GenerationResult{}, fmt.Errorf("prompt rendered to zero tokens")
	}
	cacheLen := min(r.config.MaxSeqLen, len(tokens)+options.MaxTokens+1)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, cacheLen)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	seed := options.Seed
	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
	}
	rng := NewRng(seed)
	prefillStart := time.Now()
	logits := []float32{}
	for pos, tok := range tokens {
		if pos == len(tokens)-1 {
			r.forwardTokenInto(cache, buf, tok, pos, &logits)
		} else {
			r.forwardPrefillToken(cache, buf, tok, pos)
		}
	}
	prefillTime := time.Since(prefillStart)
	decodeStart := time.Now()
	output := strings.Builder{}
	maxStopLen := 0
	for _, stop := range options.StopSequences {
		maxStopLen = max(maxStopLen, len(stop))
	}
	streamBuf := []byte{}
	flushStream := func(final bool) bool {
		if len(streamBuf) == 0 {
			return true
		}
		n := validUTF8PrefixLen(streamBuf)
		if n == 0 {
			if final {
				streamBuf = streamBuf[:0]
			}
			return true
		}
		if !final && maxStopLen > 1 {
			hold := maxStopLen - 1
			if n <= hold {
				return true
			}
			n -= hold
		}
		text := string(streamBuf[:n])
		if !onToken(text) {
			return false
		}
		copy(streamBuf, streamBuf[n:])
		streamBuf = streamBuf[:len(streamBuf)-n]
		if final && !utf8.Valid(streamBuf) {
			streamBuf = streamBuf[:0]
		}
		return true
	}
	generated := []uint32{}
	recent := append([]uint32(nil), tokens...)
	pos := len(tokens)
decode:
	for range options.MaxTokens {
		token := SampleWithScratch(logits, options.Sampler, rng, recent, &buf.SamplerCandidates)
		if r.isStopToken(token) {
			break
		}
		text := r.tok.DecodeToken(token)
		output.WriteString(text)
		current := output.String()
		if maxStopLen > 0 {
			windowStart := max(0, len(current)-maxStopLen-len(text))
			window := current[windowStart:]
			for _, stop := range options.StopSequences {
				if idx := strings.Index(window, stop); idx >= 0 {
					current = current[:windowStart+idx]
					output.Reset()
					output.WriteString(current)
					streamBuf = streamBuf[:0]
					break decode
				}
			}
		}
		streamBuf = append(streamBuf, text...)
		if !flushStream(false) {
			return GenerationResult{Text: output.String(), Stats: GenerationStats{PromptTokens: len(tokens), GeneratedTokens: len(generated), PrefillTime: prefillTime, DecodeTime: time.Since(decodeStart), TotalTime: time.Since(totalStart)}}, ErrGenerationCanceled
		}
		generated = append(generated, token)
		recent = append(recent, token)
		if len(recent) > 64 {
			recent = recent[len(recent)-64:]
		}
		if len(generated) >= options.MaxTokens || pos >= cacheLen {
			break
		}
		r.forwardTokenInto(cache, buf, token, pos, &logits)
		pos++
	}
	if !flushStream(true) {
		return GenerationResult{Text: output.String(), Stats: GenerationStats{PromptTokens: len(tokens), GeneratedTokens: len(generated), PrefillTime: prefillTime, DecodeTime: time.Since(decodeStart), TotalTime: time.Since(totalStart)}}, ErrGenerationCanceled
	}
	return GenerationResult{Text: output.String(), Stats: GenerationStats{PromptTokens: len(tokens), GeneratedTokens: len(generated), PrefillTime: prefillTime, DecodeTime: time.Since(decodeStart), TotalTime: time.Since(totalStart)}}, nil
}

func validUTF8PrefixLen(b []byte) int {
	if utf8.Valid(b) {
		return len(b)
	}
	for n := len(b) - 1; n >= 0 && len(b)-n <= utf8.UTFMax; n-- {
		if utf8.Valid(b[:n]) {
			return n
		}
	}
	return 0
}

func (r *Runner) cacheDims() (int, int, int, int, int) {
	if r.kind == loadedGemma4 {
		maxHD, maxKV, maxVal := r.config.HeadDim, r.config.NKVHeads, r.config.ValueDim
		for _, l := range r.gemma4.Layers {
			maxHD = max(maxHD, l.HeadDim)
			maxKV = max(maxKV, l.NKVHeads)
			maxVal = max(maxVal, l.ValueDim)
		}
		return maxKV * maxHD, maxKV * maxVal, maxHD, maxKV, maxVal
	}
	return r.config.NKVHeads * r.config.HeadDim, r.config.KVDim, r.config.HeadDim, r.config.NKVHeads, r.config.ValueDim
}

func (r *Runner) forwardTokenInto(cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	switch r.kind {
	case loadedGptOss:
		ForwardGptOssInto(r.config, r.gptOss, cache, buf, token, pos, logits)
	case loadedGemma4:
		ForwardGemma4Into(r.config, r.gemma4, cache, buf, token, pos, logits)
	default:
		ForwardInto(r.config, r.standard, cache, buf, token, pos, logits)
	}
}

func (r *Runner) forwardHiddenToken(cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	switch r.kind {
	case loadedGptOss:
		return ForwardHiddenGptOss(r.config, r.gptOss, cache, buf, token, pos)
	case loadedGemma4:
		return ForwardHiddenGemma4(r.config, r.gemma4, cache, buf, token, pos)
	default:
		return ForwardHidden(r.config, r.standard, cache, buf, token, pos)
	}
}

func (r *Runner) forwardPrefillToken(cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	switch r.kind {
	case loadedGptOss:
		ForwardPrefill(r.config, r.gptOss.Standard, cache, buf, token, pos)
	case loadedGemma4:
		ForwardPrefill(r.config, r.gemma4.Standard, cache, buf, token, pos)
	default:
		ForwardPrefill(r.config, r.standard, cache, buf, token, pos)
	}
}

func (r *Runner) Embed(text string) (EmbeddingResult, error) {
	r.genLock.Lock()
	defer r.genLock.Unlock()
	tokens := r.tok.Encode(text)
	if len(tokens) == 0 {
		return EmbeddingResult{}, fmt.Errorf("embed: input tokenised to zero tokens")
	}
	cacheLen := min(r.config.MaxSeqLen, len(tokens)+1)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, cacheLen)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	sum := make([]float32, r.config.Dim)
	for pos, tok := range tokens {
		h := r.forwardHiddenToken(cache, buf, tok, pos)
		for i, v := range h {
			sum[i] += v
		}
	}
	meanPoolInPlace(sum, len(tokens))
	l2NormalizeInPlace(sum)
	return EmbeddingResult{Embedding: sum, TokenCount: len(tokens)}, nil
}

func (r *Runner) isStopToken(token uint32) bool {
	if r.arch == "gpt-oss" {
		return token == r.tok.EOSID || token == 200002 || token == 200007
	}
	return token == r.tok.EOSID
}

func (r *Runner) renderMessages(messages []ChatMessage, systemPrompt string) []uint32 {
	if r.arch == "gpt-oss" {
		return r.renderGptOssMessages(messages, systemPrompt)
	}
	switch r.chatTemplateKind() {
	case "header-chat":
		if tokens, ok := r.renderHeaderChatMessages(messages, systemPrompt); ok {
			return tokens
		}
	case "chatml":
		if tokens, ok := r.renderChatMLMessages(messages, systemPrompt); ok {
			return tokens
		}
	case "phi-chat":
		if tokens, ok := r.renderPhiMessages(messages, systemPrompt); ok {
			return tokens
		}
	case "deepseek-r1-qwen":
		if tokens, ok := r.renderDeepSeekR1QwenMessages(messages, systemPrompt); ok {
			return tokens
		}
	case "granite-chat":
		if tokens, ok := r.renderGraniteMessages(messages, systemPrompt); ok {
			return tokens
		}
	}
	return r.renderPlainMessages(messages, systemPrompt)
}

func (r *Runner) renderPlainMessages(messages []ChatMessage, systemPrompt string) []uint32 {
	var b strings.Builder
	if strings.TrimSpace(systemPrompt) != "" {
		b.WriteString("System: ")
		b.WriteString(strings.TrimSpace(systemPrompt))
		b.WriteString("\n\n")
	}
	for _, m := range messages {
		switch m.Role {
		case ChatRoleSystem:
			b.WriteString("System: ")
		case ChatRoleAssistant:
			b.WriteString("Assistant: ")
		default:
			b.WriteString("User: ")
		}
		b.WriteString(strings.TrimSpace(m.Content))
		b.WriteString("\n\n")
	}
	b.WriteString("Assistant:")
	return r.tok.Encode(b.String())
}

func (r *Runner) renderGptOssMessages(messages []ChatMessage, systemPrompt string) []uint32 {
	start := specialOr(r.tok, "<|start|>", 200006)
	channel := specialOr(r.tok, "<|channel|>", 200005)
	message := specialOr(r.tok, "<|message|>", 200008)
	end := specialOr(r.tok, "<|end|>", 200007)
	user := specialOrEncoded(r.tok, "user")
	assistant := specialOrEncoded(r.tok, "assistant")
	system := specialOrEncoded(r.tok, "system")
	finalTok := specialOrEncoded(r.tok, "final")
	tokens := []uint32{}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		tokens = append(tokens, start, system, message)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(systemPrompt)...)
		tokens = append(tokens, end)
	}
	for _, m := range messages {
		role := user
		if m.Role == ChatRoleSystem {
			role = system
		} else if m.Role == ChatRoleAssistant {
			role = assistant
		}
		tokens = append(tokens, start, role, message)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(m.Content)...)
		tokens = append(tokens, end)
	}
	tokens = append(tokens, start, assistant, channel, finalTok, message)
	return tokens
}

func (r *Runner) renderHeaderChatMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	bot, ok1 := r.tok.SpecialID("<|begin_of_text|>")
	startHeader, ok2 := r.tok.SpecialID("<|start_header_id|>")
	endHeader, ok3 := r.tok.SpecialID("<|end_header_id|>")
	eot, ok4 := r.tok.SpecialID("<|eot_id|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}
	tokens := []uint32{bot}
	pushHeader := func(role string) {
		tokens = append(tokens, startHeader)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role)...)
		tokens = append(tokens, endHeader)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n\n")...)
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		pushHeader("system")
		tokens = append(tokens, r.tok.EncodeWithoutBOS(systemPrompt)...)
		tokens = append(tokens, eot)
	}
	for _, m := range messages {
		role := "user"
		if m.Role == ChatRoleSystem {
			role = "system"
		} else if m.Role == ChatRoleAssistant {
			role = "assistant"
		}
		pushHeader(role)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(m.Content)...)
		tokens = append(tokens, eot)
	}
	pushHeader("assistant")
	return tokens, true
}

func (r *Runner) renderChatMLMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	imStart, ok1 := r.tok.SpecialID("<|im_start|>")
	imEnd, ok2 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role, content string, close bool) {
		tokens = append(tokens, imStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role+"\n"+strings.TrimSpace(content))...)
		if close {
			tokens = append(tokens, imEnd)
			tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
		}
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn("system", systemPrompt, true)
	}
	for _, m := range messages {
		role := "user"
		if m.Role == ChatRoleSystem {
			role = "system"
		} else if m.Role == ChatRoleAssistant {
			role = "assistant"
		}
		appendTurn(role, m.Content, true)
	}
	tokens = append(tokens, imStart)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("assistant\n")...)
	return tokens, true
}

func (r *Runner) renderPhiMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	systemTok, ok1 := r.tok.SpecialID("<|system|>")
	userTok, ok2 := r.tok.SpecialID("<|user|>")
	assistantTok, ok3 := r.tok.SpecialID("<|assistant|>")
	endTok, ok4 := r.tok.SpecialID("<|end|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role ChatRole, content string) {
		switch role {
		case ChatRoleSystem:
			tokens = append(tokens, systemTok)
		case ChatRoleAssistant:
			tokens = append(tokens, assistantTok)
		default:
			tokens = append(tokens, userTok)
		}
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n"+strings.TrimSpace(content))...)
		tokens = append(tokens, endTok)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn(ChatRoleSystem, systemPrompt)
	}
	for _, m := range messages {
		appendTurn(m.Role, m.Content)
	}
	tokens = append(tokens, assistantTok)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	return tokens, true
}

func (r *Runner) renderDeepSeekR1QwenMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	userTok, ok1 := r.tok.SpecialID("<｜User｜>")
	assistantTok, ok2 := r.tok.SpecialID("<｜Assistant｜>")
	endTok, ok3 := r.tok.SpecialID("<｜end▁of▁sentence｜>")
	if !(ok1 && ok2 && ok3) {
		return nil, false
	}
	tokens := []uint32{r.tok.BOSID}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(systemPrompt))...)
	}
	for _, m := range messages {
		switch m.Role {
		case ChatRoleSystem:
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
		case ChatRoleAssistant:
			tokens = append(tokens, assistantTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
			tokens = append(tokens, endTok)
		default:
			tokens = append(tokens, userTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
		}
	}
	tokens = append(tokens, assistantTok)
	return tokens, true
}

func (r *Runner) renderGraniteMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	startRole, ok1 := r.tok.SpecialID("<|start_of_role|>")
	endRole, ok2 := r.tok.SpecialID("<|end_of_role|>")
	endText, ok3 := r.tok.SpecialID("<|end_of_text|>")
	if !(ok1 && ok2 && ok3) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role, content string, close bool) {
		tokens = append(tokens, startRole)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role)...)
		tokens = append(tokens, endRole)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(content))...)
		if close {
			tokens = append(tokens, endText)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(" ")...)
		}
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn("system", systemPrompt, true)
	}
	for _, m := range messages {
		role := "user"
		if m.Role == ChatRoleSystem {
			role = "system"
		} else if m.Role == ChatRoleAssistant {
			role = "assistant"
		}
		appendTurn(role, m.Content, true)
	}
	appendTurn("assistant", "", false)
	return tokens, true
}

func (r *Runner) chatTemplateKind() string {
	if v, ok := r.gguf.Metadata["tokenizer.chat_template"]; ok {
		if s, ok := v.AsString(); ok {
			switch {
			case strings.Contains(s, "<|start_header_id|>") && strings.Contains(s, "<|eot_id|>"):
				return "header-chat"
			case strings.Contains(s, "<|im_start|>") && strings.Contains(s, "<|im_end|>"):
				return "chatml"
			case strings.Contains(s, "<|user|>") && strings.Contains(s, "<|assistant|>") && strings.Contains(s, "<|end|>"):
				return "phi-chat"
			case strings.Contains(s, "<｜User｜>") && strings.Contains(s, "<｜Assistant｜>"):
				return "deepseek-r1-qwen"
			case strings.Contains(s, "<|start_of_role|>") && strings.Contains(s, "<|end_of_role|>"):
				return "granite-chat"
			}
		}
	}
	if _, ok := r.tok.SpecialID("<|im_start|>"); ok {
		if _, ok := r.tok.SpecialID("<|im_end|>"); ok {
			return "chatml"
		}
	}
	if _, ok := r.tok.SpecialID("<|user|>"); ok {
		if _, ok := r.tok.SpecialID("<|assistant|>"); ok {
			if _, ok := r.tok.SpecialID("<|end|>"); ok {
				return "phi-chat"
			}
		}
	}
	if _, ok := r.tok.SpecialID("<｜User｜>"); ok {
		if _, ok := r.tok.SpecialID("<｜Assistant｜>"); ok {
			return "deepseek-r1-qwen"
		}
	}
	if _, ok := r.tok.SpecialID("<|start_of_role|>"); ok {
		if _, ok := r.tok.SpecialID("<|end_of_role|>"); ok {
			return "granite-chat"
		}
	}
	return ""
}

func specialOr(t *Tokenizer, token string, fallback uint32) uint32 {
	if id, ok := t.SpecialID(token); ok {
		return id
	}
	return fallback
}

func specialOrEncoded(t *Tokenizer, token string) uint32 {
	if id, ok := t.SpecialID(token); ok {
		return id
	}
	ids := t.EncodeWithoutBOS(token)
	if len(ids) > 0 {
		return ids[0]
	}
	return 0
}

func meanPoolInPlace(values []float32, count int) {
	if count == 0 {
		return
	}
	scale := float32(1) / float32(count)
	for i := range values {
		values[i] *= scale
	}
}

func l2NormalizeInPlace(values []float32) {
	var ss float32
	for _, v := range values {
		ss += v * v
	}
	norm := float32(math.Sqrt(float64(ss)))
	if norm > 1e-8 {
		for i := range values {
			values[i] /= norm
		}
	}
}

func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("cosine_similarity: dimension mismatch (%d vs %d)", len(a), len(b))
	}
	if len(a) == 0 {
		return 0, fmt.Errorf("cosine_similarity: empty vectors")
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := float32(math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
	if denom <= 1e-12 {
		return 0, fmt.Errorf("cosine_similarity: zero-norm vector encountered")
	}
	return dot / denom, nil
}
