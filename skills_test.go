package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadSkillsEmptyDirIsNoOp(t *testing.T) {
	skills, err := LoadSkills("")
	if err != nil || skills != nil {
		t.Fatalf("skills=%v err=%v, want nil,nil", skills, err)
	}
}

func TestLoadSkillsPerDirectoryLayout(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "pdf-fill", "SKILL.md"), "---\nname: pdf-fill\ndescription: Fill out a PDF form.\n---\nFull instructions here.\n")
	writeFile(t, filepath.Join(dir, "git-review", "SKILL.md"), "---\nname: git-review\ndescription: Review a git diff.\n---\nStep 1. Step 2.\n")

	skills, err := LoadSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(skills) != 2 {
		t.Fatalf("len = %d, want 2", len(skills))
	}
	// Sorted by name.
	if skills[0].Name != "git-review" || skills[1].Name != "pdf-fill" {
		t.Fatalf("names = %q, %q", skills[0].Name, skills[1].Name)
	}
	if skills[1].Description != "Fill out a PDF form." {
		t.Fatalf("description = %q", skills[1].Description)
	}
	if skills[1].Body != "Full instructions here." {
		t.Fatalf("body = %q", skills[1].Body)
	}
}

func TestLoadSkillsFlatMarkdownLayout(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "quickstart.md"), "---\ndescription: A quick primer.\n---\nBody text.\n")

	skills, err := LoadSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(skills) != 1 || skills[0].Name != "quickstart" {
		t.Fatalf("skills = %+v", skills)
	}
	if skills[0].Description != "A quick primer." {
		t.Fatalf("description = %q", skills[0].Description)
	}
}

func TestLoadSkillsIgnoresUnrelatedFiles(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "README.txt"), "not a skill")
	writeFile(t, filepath.Join(dir, "empty-dir", "notes.txt"), "no SKILL.md here")
	writeFile(t, filepath.Join(dir, "real", "SKILL.md"), "---\nname: real\ndescription: d\n---\nb")

	skills, err := LoadSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(skills) != 1 || skills[0].Name != "real" {
		t.Fatalf("skills = %+v", skills)
	}
}

func TestLoadSkillsMissingDirErrors(t *testing.T) {
	if _, err := LoadSkills(filepath.Join(t.TempDir(), "does-not-exist")); err == nil {
		t.Fatal("expected an error for a missing directory")
	}
}

func TestParseFrontmatterNoHeaderIsAllBody(t *testing.T) {
	meta, body := parseFrontmatter("just a plain markdown file\nwith no frontmatter")
	if len(meta) != 0 {
		t.Fatalf("meta = %v, want empty", meta)
	}
	if body != "just a plain markdown file\nwith no frontmatter" {
		t.Fatalf("body = %q", body)
	}
}

func TestParseFrontmatterQuotedValues(t *testing.T) {
	meta, _ := parseFrontmatter("---\nname: \"quoted-name\"\ndescription: 'single quoted'\n---\nbody")
	if meta["name"] != "quoted-name" || meta["description"] != "single quoted" {
		t.Fatalf("meta = %v", meta)
	}
}

func TestFindSkill(t *testing.T) {
	skills := []Skill{{Name: "a", Description: "da"}, {Name: "b", Description: "db"}}
	if s, ok := findSkill(skills, "b"); !ok || s.Description != "db" {
		t.Fatalf("findSkill(b) = %+v, %v", s, ok)
	}
	if _, ok := findSkill(skills, "missing"); ok {
		t.Fatal("expected not found")
	}
}

func TestSkillsToolDefinition(t *testing.T) {
	skills := []Skill{{Name: "pdf-fill", Description: "Fill a PDF."}, {Name: "git-review", Description: "Review a diff."}}
	def := skillsToolDefinition(skills)
	if def.Function.Name != LoadSkillToolName {
		t.Fatalf("name = %q", def.Function.Name)
	}
	if !strings.Contains(def.Function.Description, "pdf-fill") || !strings.Contains(def.Function.Description, "Fill a PDF.") {
		t.Fatalf("description missing skill listing: %q", def.Function.Description)
	}
	if !strings.Contains(string(def.Function.Parameters), `"pdf-fill"`) || !strings.Contains(string(def.Function.Parameters), `"git-review"`) {
		t.Fatalf("parameters missing enum values: %s", def.Function.Parameters)
	}
}
