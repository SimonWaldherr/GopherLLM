package gopherllm

import (
	"bytes"
	"strings"
	"testing"
)

func TestDefaultPrivacyReportIsLocalAndTelemetryFree(t *testing.T) {
	report := DefaultPrivacyReport()
	if report.Telemetry || report.DefaultNetwork != "none" || !strings.Contains(report.Inference, "local") {
		t.Fatalf("unexpected report: %#v", report)
	}
	var out bytes.Buffer
	if err := WritePrivacyReport(&out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"telemetry": false`, "OpenStreetMap place search", "Hugging Face import"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("report missing %q: %s", want, out.String())
		}
	}
}
