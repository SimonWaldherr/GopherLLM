package hestia

import (
	"embed"
	"io/fs"
)

//go:embed web/index.html
var webEmbed embed.FS

// webFS is rooted at web/ (not the embed.FS's own root, which still
// carries the "web/" prefix from the source file's directory) so
// http.FileServer serves "/" as index.html rather than needing "/web/".
var webFS = mustSub(webEmbed, "web")

func mustSub(f embed.FS, dir string) fs.FS {
	sub, err := fs.Sub(f, dir)
	if err != nil {
		panic(err)
	}
	return sub
}
