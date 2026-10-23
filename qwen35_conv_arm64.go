//go:build arm64

package gopherllm

import "unsafe"

//go:noescape
func qwen35Conv4NEON(kernel, history, values *float32, channels int)

// qwen35Conv4 applies a four-tap causal depthwise convolution in place.
// Kernel rows contain four weights and history rows contain three preceding
// inputs. The caller selects SSMConv=4/ConvLen=3 and passes one layer's history;
// all three slices must occupy separate storage. A false result leaves them
// unchanged so the caller can use the general convolution.
func qwen35Conv4(kernel, history, values []float32) bool {
	n := len(values)
	if n < 4 || len(kernel)/4 < n || len(history)/3 < n {
		return false
	}
	n4 := n &^ 3
	qwen35Conv4NEON(unsafe.SliceData(kernel), unsafe.SliceData(history), unsafe.SliceData(values), n4)
	for ch := n4; ch < n; ch++ {
		k, h := kernel[ch*4:ch*4+4], history[ch*3:ch*3+3]
		input := values[ch]
		v := float32(0)
		for i := range 3 {
			v += k[i] * h[i]
		}
		v += k[3] * input
		h[0], h[1], h[2] = h[1], h[2], input
		values[ch] = v
	}
	return true
}
