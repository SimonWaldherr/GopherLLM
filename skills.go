package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Skills give a model on-demand access to written-out instructions it doesn't
// need in context by default: a name and one-line description are always
// visible (via the load_skill tool's description), and the full body is only
// loaded when the model actually asks for it — the same progressive-disclosure
// design as Claude's Agent Skills. Layout on disk mirrors that convention too:
//
//	skills/
//	  pdf-fill/SKILL.md
//	  git-review/SKILL.md
//
// each SKILL.md is a Markdown file with a YAML-ish frontmatter header:
//
//	---
//	name: pdf-fill
//	description: Fill out a PDF form given field values.
//	---
//	Full instructions the model receives once it loads this skill...
//
// A flat "skills/*.md" layout (no per-skill subdirectory) is also accepted,
// using the filename (minus extension) as the fallback name.
type Skill struct {
	Name        string
	Description string
	Body        string
	Path        string
}

// LoadSkillToolName is the name of the tool RunAgenticChat resolves internally
// (see agent.go) rather than returning to the caller.
const LoadSkillToolName = "load_skill"

// LoadSkills discovers every skill under dir. An empty dir returns (nil, nil)
// so the feature is a no-op unless explicitly configured.
func LoadSkills(dir string) ([]Skill, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("skills: %w", err)
	}
	var skills []Skill
	for _, e := range entries {
		var path, fallbackName string
		switch {
		case e.IsDir():
			candidate := filepath.Join(dir, e.Name(), "SKILL.md")
			if _, statErr := os.Stat(candidate); statErr != nil {
				continue
			}
			path, fallbackName = candidate, e.Name()
		case strings.EqualFold(filepath.Ext(e.Name()), ".md"):
			path = filepath.Join(dir, e.Name())
			fallbackName = strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))
		default:
			continue
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return nil, fmt.Errorf("skills: reading %s: %w", path, readErr)
		}
		meta, body := parseFrontmatter(string(data))
		name := meta["name"]
		if name == "" {
			name = fallbackName
		}
		skills = append(skills, Skill{Name: name, Description: meta["description"], Body: strings.TrimSpace(body), Path: path})
	}
	sort.Slice(skills, func(i, j int) bool { return skills[i].Name < skills[j].Name })
	return skills, nil
}

// parseFrontmatter splits a "---\nkey: value\n---\nbody" document into its
// header fields and body. Documents without a recognizable frontmatter block
// are returned as an all-body document with an empty header, rather than an
// error, so a plain (non-frontmatter) Markdown file still loads as a skill
// with a body but no declared name/description.
func parseFrontmatter(content string) (meta map[string]string, body string) {
	meta = map[string]string{}
	if !strings.HasPrefix(content, "---") {
		return meta, content
	}
	rest := strings.TrimPrefix(strings.TrimPrefix(content[3:], "\r\n"), "\n")
	end := strings.Index(rest, "\n---")
	if end < 0 {
		return meta, content
	}
	header, afterMarker := rest[:end], rest[end+len("\n---"):]
	if nl := strings.IndexByte(afterMarker, '\n'); nl >= 0 {
		body = afterMarker[nl+1:]
	}
	for _, line := range strings.Split(header, "\n") {
		k, v, ok := strings.Cut(strings.TrimRight(line, "\r"), ":")
		if !ok {
			continue
		}
		meta[strings.TrimSpace(k)] = strings.Trim(strings.TrimSpace(v), `"'`)
	}
	return meta, body
}

// findSkill looks up a skill by exact name.
func findSkill(skills []Skill, name string) (Skill, bool) {
	for _, s := range skills {
		if s.Name == name {
			return s, true
		}
	}
	return Skill{}, false
}

// skillsToolDefinition synthesizes the load_skill tool offered to the model
// whenever skills are configured: every name+description is listed in the
// tool's own description (so the model can decide when a skill is relevant
// without spending context on its full body), and the name parameter is
// constrained to the known skill names via a JSON-schema enum.
func skillsToolDefinition(skills []Skill) ToolDefinition {
	names := make([]string, len(skills))
	var listing strings.Builder
	for i, s := range skills {
		names[i] = s.Name
		fmt.Fprintf(&listing, "- %s: %s\n", s.Name, s.Description)
	}
	enumJSON, _ := json.Marshal(names)
	nameJSON, _ := json.Marshal(fmt.Sprintf("The exact name of the skill to load. One of: %s", strings.Join(names, ", ")))
	params := fmt.Sprintf(`{"type":"object","properties":{"name":{"type":"string","enum":%s,"description":%s}},"required":["name"]}`, enumJSON, nameJSON)
	return ToolDefinition{
		Type: "function",
		Function: ToolFunctionDef{
			Name:        LoadSkillToolName,
			Description: "Load the full instructions for a named skill before using it, then follow them. Available skills:\n" + listing.String(),
			Parameters:  json.RawMessage(params),
		},
	}
}
