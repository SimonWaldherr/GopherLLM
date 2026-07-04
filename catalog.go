package gopherllm

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const lmStudioCommunitySubdir = ".cache/lm-studio/models/lmstudio-community"

// ModelEntry describes one GGUF file found under the model directory. ID is
// the root-relative path without the .gguf suffix (unique within a scan and
// what --list-models prints); IsProjector marks multimodal companion files
// (mmproj-*/clip) that must not be offered for text generation.
type ModelEntry struct {
	ID           string
	Repository   string
	FileName     string
	Path         string
	SizeBytes    int64
	Architecture string
	ModelName    string
	IsProjector  bool
	IsSupported  bool
}

func (m ModelEntry) Status() string {
	if m.IsProjector {
		return "projector"
	}
	if m.IsSupported {
		return "supported"
	}
	return "unsupported"
}

// DefaultModelDir returns the directory scanned when --model-dir is not
// given: $RUSTY_LLM_MODEL_DIR if set, else the LM Studio community models
// directory under $HOME. (MODEL_DIR is a Makefile variable, not read here.)
func DefaultModelDir() string {
	if path := strings.TrimSpace(os.Getenv("RUSTY_LLM_MODEL_DIR")); path != "" {
		return path
	}
	if home := os.Getenv("HOME"); home != "" {
		return filepath.Join(home, lmStudioCommunitySubdir)
	}
	return lmStudioCommunitySubdir
}

// DiscoverModels recursively finds every .gguf under root and parses each
// file's header (mmap'd, weights untouched) to fill in architecture, name,
// and support status. Unparseable files are skipped with a stderr note rather
// than failing the whole scan. Results are sorted by ID.
func DiscoverModels(root string) ([]ModelEntry, error) {
	files := []string{}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.EqualFold(filepath.Ext(path), ".gguf") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to scan %s: %w", root, err)
	}
	sort.Strings(files)
	entries := make([]ModelEntry, 0, len(files))
	for _, path := range files {
		entry, err := inspectModel(root, path)
		if err != nil {
			fmt.Fprintf(stderr(), "Skipping %s: %v\n", path, err)
			continue
		}
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].ID < entries[j].ID })
	return entries, nil
}

// ResolveModelPath turns the CLI's model selector into a concrete .gguf path.
// A selector that is an existing file wins outright; an existing directory
// (or a nil selector) opens the interactive picker over that directory;
// anything else is matched against the discovered models by SelectModel.
func ResolveModelPath(selection *string, modelDir string) (string, error) {
	if selection != nil {
		selected := *selection
		if st, err := os.Stat(selected); err == nil {
			if st.Mode().IsRegular() {
				return selected, nil
			}
			if st.IsDir() {
				return chooseFromDirectory(selected, nil, os.Stdin, stderr())
			}
			return "", fmt.Errorf("model path is neither a file nor a directory: %s", selected)
		}
		entries, err := DiscoverModels(modelDir)
		if err != nil {
			return "", err
		}
		entry, err := SelectModel(entries, selected)
		if err != nil {
			return "", err
		}
		return entry.Path, nil
	}
	return chooseFromDirectory(modelDir, nil, os.Stdin, stderr())
}

// SelectModel matches selector against the usable (supported, non-projector)
// entries: exact matches on id/repo/filename/name/path beat substring
// matches, a unique match wins, and ambiguity or matching only an
// unsupported/projector file produces a descriptive error listing the
// choices.
func SelectModel(entries []ModelEntry, selector string) (ModelEntry, error) {
	selector = strings.TrimSpace(selector)
	if selector == "" {
		return ModelEntry{}, fmt.Errorf("model selector must not be empty")
	}
	usable := []ModelEntry{}
	for _, e := range entries {
		if e.IsSupported && !e.IsProjector {
			usable = append(usable, e)
		}
	}
	matches := matchingEntries(usable, selector)
	if len(matches) == 1 {
		return matches[0], nil
	}
	if len(matches) > 1 {
		return ModelEntry{}, fmt.Errorf("%s", formatAmbiguous(selector, matches))
	}
	unsupported := matchingEntries(entries, selector)
	if len(unsupported) == 1 {
		e := unsupported[0]
		arch := e.Architecture
		if arch == "" {
			arch = "unknown"
		}
		return ModelEntry{}, fmt.Errorf("model %q matched %s, but it is marked as %s (architecture: %s)", selector, e.ID, e.Status(), arch)
	}
	if len(unsupported) > 1 {
		return ModelEntry{}, fmt.Errorf("%s", formatAmbiguous(selector, unsupported))
	}
	return ModelEntry{}, fmt.Errorf("no GGUF model matched %q; use --list-models to see available models in %s", selector, modelDirFromEntries(entries))
}

func PrintModelList(entries []ModelEntry) {
	if len(entries) == 0 {
		fmt.Println("No GGUF files found.")
		return
	}
	fmt.Printf("%-62s %-14s %-12s size\n", "id", "architecture", "status")
	for _, e := range entries {
		arch := e.Architecture
		if arch == "" {
			arch = "unknown"
		}
		fmt.Printf("%-62s %-14s %-12s %.2f GB\n", truncate(e.ID, 62), truncate(arch, 14), e.Status(), float64(e.SizeBytes)/(1024*1024*1024))
	}
}

func chooseFromDirectory(dir string, selector *string, in io.Reader, out io.Writer) (string, error) {
	entries, err := DiscoverModels(dir)
	if err != nil {
		return "", err
	}
	if selector != nil {
		entry, err := SelectModel(entries, *selector)
		if err != nil {
			return "", err
		}
		return entry.Path, nil
	}
	usable := []ModelEntry{}
	for _, e := range entries {
		if e.IsSupported && !e.IsProjector {
			usable = append(usable, e)
		}
	}
	switch len(usable) {
	case 0:
		return "", fmt.Errorf("no supported text GGUF models found in %s", dir)
	case 1:
		return usable[0].Path, nil
	default:
		entry, err := PromptModelSelection(dir, usable, in, out)
		if err != nil {
			return "", err
		}
		return entry.Path, nil
	}
}

// PromptModelSelection runs the interactive terminal picker: a numbered menu
// that accepts an index, a filter substring (re-listing the matches), or
// q/quit/exit to abort.
func PromptModelSelection(dir string, entries []ModelEntry, in io.Reader, out io.Writer) (ModelEntry, error) {
	if len(entries) == 0 {
		return ModelEntry{}, fmt.Errorf("no selectable GGUF models found in %s", dir)
	}
	if len(entries) == 1 {
		return entries[0], nil
	}
	if out == nil {
		out = io.Discard
	}
	if in == nil {
		return ModelEntry{}, fmt.Errorf("multiple GGUF models found in %s, but no input is available", dir)
	}

	fmt.Fprintf(out, "\nFound %d supported GGUF models in %s:\n\n", len(entries), dir)
	printModelMenu(entries, out, len(entries))
	fmt.Fprintln(out)
	fmt.Fprintln(out, "Select a model by number, enter text to filter, or type q to quit.")

	scanner := bufio.NewScanner(in)
	for {
		fmt.Fprint(out, "model> ")
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return ModelEntry{}, err
			}
			return ModelEntry{}, fmt.Errorf("model selection aborted")
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if strings.EqualFold(input, "q") || strings.EqualFold(input, "quit") || strings.EqualFold(input, "exit") {
			return ModelEntry{}, fmt.Errorf("model selection aborted")
		}
		if idx, err := strconv.Atoi(input); err == nil {
			if idx >= 1 && idx <= len(entries) {
				selected := entries[idx-1]
				fmt.Fprintf(out, "Selected: %s\n\n", selected.ID)
				return selected, nil
			}
			fmt.Fprintf(out, "Please enter a number between 1 and %d.\n", len(entries))
			continue
		}

		matches := matchingEntries(entries, input)
		switch len(matches) {
		case 0:
			fmt.Fprintf(out, "No model matched %q.\n", input)
		case 1:
			fmt.Fprintf(out, "Selected: %s\n\n", matches[0].ID)
			return matches[0], nil
		default:
			fmt.Fprintf(out, "\n%d matches for %q:\n\n", len(matches), input)
			printModelMenuWithOriginalNumbers(entries, matches, out, min(len(matches), 20))
			if len(matches) > 20 {
				fmt.Fprintf(out, "  ... %d more matches. Refine the filter.\n", len(matches)-20)
			}
			fmt.Fprintln(out)
			fmt.Fprintln(out, "Enter a number from the original list or refine the filter.")
		}
	}
}

func printModelMenu(entries []ModelEntry, out io.Writer, limit int) {
	for i, e := range entries[:limit] {
		printModelMenuLine(out, i+1, e)
	}
}

func printModelMenuWithOriginalNumbers(all []ModelEntry, entries []ModelEntry, out io.Writer, limit int) {
	for _, e := range entries[:limit] {
		printModelMenuLine(out, modelMenuIndex(all, e), e)
	}
}

func printModelMenuLine(out io.Writer, index int, e ModelEntry) {
	arch := e.Architecture
	if arch == "" {
		arch = "unknown"
	}
	modelName := e.ModelName
	if modelName == "" {
		modelName = e.FileName
	}
	fmt.Fprintf(
		out,
		"  %2d. %-62s %-14s %6.2f GB  %s\n",
		index,
		truncate(e.ID, 62),
		truncate(arch, 14),
		float64(e.SizeBytes)/(1024*1024*1024),
		truncate(modelName, 72),
	)
}

func modelMenuIndex(entries []ModelEntry, entry ModelEntry) int {
	for i, e := range entries {
		if e.Path == entry.Path && e.ID == entry.ID {
			return i + 1
		}
	}
	return 0
}

func inspectModel(root, path string) (ModelEntry, error) {
	st, err := os.Stat(path)
	if err != nil {
		return ModelEntry{}, err
	}
	mmap, err := OpenMmap(path)
	if err != nil {
		return ModelEntry{}, err
	}
	defer mmap.Close()
	gguf, err := ParseGGUFQuiet(mmap.Bytes())
	if err != nil {
		return ModelEntry{}, err
	}
	fileName := filepath.Base(path)
	repository := filepath.Base(filepath.Dir(path))
	rel, err := filepath.Rel(root, path)
	id := strings.TrimSuffix(fileName, ".gguf")
	if err == nil {
		id = strings.TrimSuffix(rel, ".gguf")
	}
	arch, _ := gguf.GetString("general.architecture")
	modelName, _ := gguf.GetString("general.name")
	lower := strings.ToLower(fileName)
	isProjector := strings.HasPrefix(lower, "mmproj-") || strings.Contains(lower, "mmproj") || strings.EqualFold(arch, "clip")
	return ModelEntry{ID: id, Repository: repository, FileName: fileName, Path: path, SizeBytes: st.Size(), Architecture: arch, ModelName: modelName, IsProjector: isProjector, IsSupported: ArchitectureSupported(arch)}, nil
}

func matchingEntries(entries []ModelEntry, selector string) []ModelEntry {
	needle := strings.ToLower(selector)
	exact := []ModelEntry{}
	partial := []ModelEntry{}
	for _, e := range entries {
		keys := []string{e.ID, e.Repository, e.FileName, e.ModelName, e.Path}
		foundExact, foundPartial := false, false
		for _, key := range keys {
			if strings.EqualFold(key, selector) {
				foundExact = true
				break
			}
			if strings.Contains(strings.ToLower(key), needle) {
				foundPartial = true
			}
		}
		if foundExact {
			exact = append(exact, e)
		} else if foundPartial {
			partial = append(partial, e)
		}
	}
	if len(exact) > 0 {
		return exact
	}
	return partial
}

func formatAmbiguous(selector string, entries []ModelEntry) string {
	return fmt.Sprintf("model selector %q matched multiple GGUF files:\n\n%s", selector, formatModelChoices(entries))
}

func formatModelChoices(entries []ModelEntry) string {
	lines := make([]string, len(entries))
	for i, e := range entries {
		arch := e.Architecture
		if arch == "" {
			arch = "unknown"
		}
		lines[i] = fmt.Sprintf("  - %s [%s; %s]", e.ID, arch, e.Status())
	}
	return strings.Join(lines, "\n")
}

func modelDirFromEntries(entries []ModelEntry) string {
	if len(entries) == 0 {
		return "the model directory"
	}
	return filepath.Dir(entries[0].Path)
}

func truncate(value string, maxChars int) string {
	r := []rune(value)
	if len(r) <= maxChars {
		return value
	}
	if maxChars <= 1 {
		return "~"
	}
	return string(r[:maxChars-1]) + "~"
}
