package main

import "testing"

func TestGGMLTypeBlockBytes(t *testing.T) {
	tests := []struct {
		typ  GGMLType
		want int
	}{
		{GGMLTypeF32, 4},
		{GGMLTypeF16, 2},
		{GGMLTypeQ4_0, 18},
		{GGMLTypeQ4_1, 20},
		{GGMLTypeQ5_0, 22},
		{GGMLTypeQ5_1, 24},
		{GGMLTypeQ8_0, 34},
		{GGMLTypeQ8_1, 36},
	}
	for _, tt := range tests {
		got, ok := tt.typ.BlockBytes()
		if !ok || got != tt.want {
			t.Fatalf("%s BlockBytes = %d, %v; want %d, true", tt.typ, got, ok, tt.want)
		}
	}
}
