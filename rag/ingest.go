package rag

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"path"
	"strings"
)

// IngestOptions configures AddFS's directory walk.
type IngestOptions struct {
	// Match selects which files to read. The default accepts common
	// plain-text extensions (.txt .md .markdown .csv .tsv .json .log .go and
	// a handful of others); a file is skipped when Match returns false.
	Match func(path string, info fs.FileInfo) bool
	// MaxFileBytes skips a file larger than this. Zero means 4 MiB.
	MaxFileBytes int64
	// FollowSymlinks controls whether a symlink entry is followed. Default
	// false: an untrusted or accidentally-circular directory tree should not
	// be walked into just because AddFS was pointed at it.
	FollowSymlinks bool
	// Extract turns a file's raw bytes into a Doc. The default treats the
	// content as UTF-8 text and uses the file's base name as Doc.Title. This
	// is the hook for PDF/DOCX/HTML ingestion: this package ships none of
	// those extractors (see the package doc's non-goals) — bring your own.
	Extract func(path string, data []byte) (Doc, error)
}

func (o IngestOptions) maxFileBytes() int64 {
	if o.MaxFileBytes > 0 {
		return o.MaxFileBytes
	}
	return 4 << 20
}

func (o IngestOptions) match() func(string, fs.FileInfo) bool {
	if o.Match != nil {
		return o.Match
	}
	return defaultIngestMatch
}

var defaultIngestExtensions = map[string]bool{
	".txt": true, ".md": true, ".markdown": true, ".csv": true, ".tsv": true,
	".json": true, ".log": true, ".go": true, ".yaml": true, ".yml": true,
	".rst": true, ".org": true,
}

func defaultIngestMatch(p string, info fs.FileInfo) bool {
	if info.IsDir() {
		return false
	}
	return defaultIngestExtensions[strings.ToLower(path.Ext(p))]
}

func (o IngestOptions) extract() func(string, []byte) (Doc, error) {
	if o.Extract != nil {
		return o.Extract
	}
	return defaultExtract
}

func defaultExtract(filePath string, data []byte) (Doc, error) {
	return Doc{ID: filePath, Title: path.Base(filePath), Path: filePath, Text: string(data)}, nil
}

// AddFS walks fsys and indexes every file opts.Match accepts (or the default
// plain-text extension list), using opts.Extract (or a plain UTF-8 read) to
// turn each into a Doc. added and skipped count files, not chunks.
//
// A file Extract errors on, or one over MaxFileBytes, is skipped and does not
// fail the walk. The first error from the walk itself (a permission failure,
// an unreadable directory) does fail it, since that indicates fsys cannot be
// trusted rather than one file being unreadable content.
func (ix *Index) AddFS(ctx context.Context, fsys fs.FS, opts IngestOptions) (added, skipped int, err error) {
	match := opts.match()
	extract := opts.extract()
	maxBytes := opts.maxFileBytes()

	var docs []Doc
	walkErr := fs.WalkDir(fsys, ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if !opts.FollowSymlinks && d.Type()&fs.ModeSymlink != 0 {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if !match(p, info) {
			return nil
		}
		if info.Size() > maxBytes {
			skipped++
			return nil
		}
		f, err := fsys.Open(p)
		if err != nil {
			return err
		}
		data, err := io.ReadAll(io.LimitReader(f, maxBytes+1))
		f.Close()
		if err != nil {
			return err
		}
		if int64(len(data)) > maxBytes {
			skipped++
			return nil
		}
		doc, err := extract(p, data)
		if err != nil {
			skipped++
			return nil
		}
		docs = append(docs, doc)
		return nil
	})
	if walkErr != nil {
		return added, skipped, fmt.Errorf("rag: walk: %w", walkErr)
	}
	if err := ix.Add(ctx, docs...); err != nil {
		return added, skipped, err
	}
	return len(docs), skipped, nil
}
