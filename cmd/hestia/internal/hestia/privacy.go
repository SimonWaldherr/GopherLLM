package hestia

import (
	"net"
	"net/http"
	"strings"
)

func localOnly(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Permissions-Policy", "microphone=(self), camera=(), geolocation=()")
		w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; worker-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")

		remote, _, err := net.SplitHostPort(r.RemoteAddr)
		host, _, hostErr := net.SplitHostPort(r.Host)
		if hostErr != nil {
			host = strings.TrimPrefix(strings.TrimSuffix(r.Host, "]"), "[")
		}
		if err != nil || !loopbackHost(remote) || !loopbackHost(host) {
			writeError(w, http.StatusForbidden, jsonErr("hestia: loopback access only"))
			return
		}
		scheme := "http"
		if r.TLS != nil {
			scheme = "https"
		}
		origin := r.Header.Get("Origin")
		fetchSite := r.Header.Get("Sec-Fetch-Site")
		if (origin != "" && !strings.EqualFold(origin, scheme+"://"+r.Host)) || fetchSite == "cross-site" || fetchSite == "same-site" {
			writeError(w, http.StatusForbidden, jsonErr("hestia: cross-origin access denied"))
			return
		}
		next.ServeHTTP(w, r)
	})
}
