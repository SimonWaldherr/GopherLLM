package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFacadePreservesFunctionDeclarations(t *testing.T) {
	t.Chdir(t.TempDir())
	if err := os.MkdirAll("internal/engine", 0755); err != nil {
		t.Fatal(err)
	}
	source := `package engine
import "context"
// Model holds state.
type Model struct{}
const Mode = 1
var ErrClosed error
func New[T any]() T { panic("fixture") }
func Stream(ctx context.Context, values ...string) error { return nil }
func Unnamed(context.Context, int) string { return "" }
func Ignored(_ int) {}
func (m *Model) Close() error { return nil }
func private() {}
`
	if err := os.WriteFile(filepath.Join("internal/engine", "api.go"), []byte(source), 0644); err != nil {
		t.Fatal(err)
	}
	if err := generate(false); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("api_generated.go")
	if err != nil {
		t.Fatal(err)
	}
	output := string(data)
	for _, want := range []string{
		"// Model holds state.\ntype Model = engineimpl.Model",
		"const Mode = engineimpl.Mode", "var ErrClosed = engineimpl.ErrClosed",
		"func New[T any]() T", "return engineimpl.New[T]()",
		"func Stream(ctx context.Context, values ...string) error",
		"return engineimpl.Stream(ctx, values...)",
		"func Unnamed(arg0 context.Context, arg1 int) string",
		"return engineimpl.Unnamed(arg0, arg1)",
		"func Ignored(arg0 int)", "engineimpl.Ignored(arg0)",
	} {
		if !strings.Contains(output, want) {
			t.Errorf("facade missing %q", want)
		}
	}
	if strings.Contains(output, "private") || strings.Contains(output, "func (m") {
		t.Fatal("facade must rely on aliases for methods and omit private declarations")
	}
	if err := generate(true); err != nil {
		t.Fatalf("fresh facade rejected: %v", err)
	}
	if err := os.WriteFile("api_generated.go", []byte(strings.ReplaceAll(output, "\n", "\r\n")), 0644); err != nil {
		t.Fatal(err)
	}
	if err := generate(true); err != nil {
		t.Fatalf("Windows line endings rejected: %v", err)
	}
	if err := os.WriteFile("api_generated.go", []byte("stale"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := generate(true); err == nil {
		t.Fatal("check accepted a stale facade")
	}
	data, err = os.ReadFile("api_generated.go")
	if err != nil || string(data) != "stale" {
		t.Fatalf("check modified the facade: %q, %v", data, err)
	}
}
