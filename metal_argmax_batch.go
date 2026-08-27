package gopherllm

// metalBatchArgmaxMaxTokens is the bounded draft window accepted by the
// GPU-resident Q6_K vocabulary projection. It intentionally covers the useful
// greedy speculative range without retaining arbitrary [batch][vocab] output
// storage in the process-wide Metal workspace.
const metalBatchArgmaxMaxTokens = 8
