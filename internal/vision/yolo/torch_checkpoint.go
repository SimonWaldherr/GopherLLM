package yolo

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/internal/numeric"
)

// This file reads PyTorch's torch.save checkpoint format (a zip archive
// holding one data.pkl pickle plus one raw file per tensor storage) far
// enough to recover tensors, without Python and without executing anything.
//
// Pickle is a small stack-machine language whose GLOBAL/REDUCE/BUILD
// opcodes normally import and call arbitrary Python callables, which is why
// loading an untrusted .pt with pickle.load is remote code execution. The
// interpreter below never calls anything: a GLOBAL becomes an inert
// (module, name) value, and REDUCE/NEWOBJ/BUILD only record their arguments,
// except for a fixed allowlist of pure data constructors (tensor and
// parameter rebuilds, OrderedDict, set, torch.Size) that it reproduces
// itself. The result is plain data — dicts, lists, numbers, strings and
// tensor references — that callers walk to find the weights they need.
//
// The zip reader is likewise minimal on purpose: torch.save writes every
// entry uncompressed ("stored"), so reading one needs only the central
// directory and no compress/flate or hash/crc32, which keeps this package's
// dependency closure unchanged (see TestImageDecodersAreDroppable).

// torchZip indexes the stored entries of a torch.save archive.
type torchZip struct {
	data    []byte
	entries map[string][]byte
}

func openTorchZip(data []byte) (*torchZip, error) {
	const eocdSig, eocdLen = 0x06054b50, 22
	eocd := -1
	for i := len(data) - eocdLen; i >= 0 && i >= len(data)-eocdLen-0xffff; i-- {
		if binary.LittleEndian.Uint32(data[i:]) == eocdSig {
			eocd = i
			break
		}
	}
	if eocd < 0 {
		return nil, errors.New("not a zip archive (no end-of-central-directory record)")
	}
	count := uint64(binary.LittleEndian.Uint16(data[eocd+10:]))
	dirOffset := uint64(binary.LittleEndian.Uint32(data[eocd+16:]))
	// ZIP64: a locator right before the classic record points at the real one.
	if loc := eocd - 20; loc >= 0 && binary.LittleEndian.Uint32(data[loc:]) == 0x07064b50 {
		rec := binary.LittleEndian.Uint64(data[loc+8:])
		if rec > uint64(len(data)) || uint64(len(data))-rec < 56 || binary.LittleEndian.Uint32(data[rec:]) != 0x06064b50 {
			return nil, errors.New("zip64 end-of-central-directory record is missing")
		}
		count = binary.LittleEndian.Uint64(data[rec+32:])
		dirOffset = binary.LittleEndian.Uint64(data[rec+48:])
	}
	// fits reports whether [off, off+n) lies within data, without the
	// overflow a plain off+n > len comparison allows for hostile offsets.
	size64 := uint64(len(data))
	fits := func(off, n uint64) bool { return off <= size64 && n <= size64-off }
	z := &torchZip{data: data, entries: map[string][]byte{}}
	p := dirOffset
	for range count {
		if !fits(p, 46) || binary.LittleEndian.Uint32(data[p:]) != 0x02014b50 {
			return nil, errors.New("corrupt zip central directory")
		}
		method := binary.LittleEndian.Uint16(data[p+10:])
		size := uint64(binary.LittleEndian.Uint32(data[p+20:]))
		nameLen := uint64(binary.LittleEndian.Uint16(data[p+28:]))
		extraLen := uint64(binary.LittleEndian.Uint16(data[p+30:]))
		commentLen := uint64(binary.LittleEndian.Uint16(data[p+32:]))
		local := uint64(binary.LittleEndian.Uint32(data[p+42:]))
		if !fits(p+46, nameLen+extraLen) {
			return nil, errors.New("corrupt zip central directory entry")
		}
		name := string(data[p+46 : p+46+nameLen])
		if size == math.MaxUint32 || local == math.MaxUint32 {
			// The ZIP64 extra field lists, in order, only the values whose
			// classic field is saturated: uncompressed, compressed, offset.
			extra := data[p+46+nameLen : p+46+nameLen+extraLen]
			for len(extra) >= 4 {
				id, n := binary.LittleEndian.Uint16(extra), int(binary.LittleEndian.Uint16(extra[2:]))
				if 4+n > len(extra) {
					break
				}
				if id == 1 {
					field := extra[4 : 4+n]
					next := func() uint64 {
						if len(field) < 8 {
							return 0
						}
						v := binary.LittleEndian.Uint64(field)
						field = field[8:]
						return v
					}
					if size == math.MaxUint32 {
						size = next()
						next() // compressed size equals size for stored entries
					}
					if local == math.MaxUint32 {
						local = next()
					}
				}
				extra = extra[4+n:]
			}
		}
		p += 46 + nameLen + extraLen + commentLen
		if strings.HasSuffix(name, "/") {
			continue
		}
		if method != 0 {
			return nil, fmt.Errorf("zip entry %s is compressed (method %d); torch.save archives store entries uncompressed", name, method)
		}
		if !fits(local, 30) || binary.LittleEndian.Uint32(data[local:]) != 0x04034b50 {
			return nil, fmt.Errorf("zip entry %s has no local header", name)
		}
		start := local + 30 + uint64(binary.LittleEndian.Uint16(data[local+26:])) + uint64(binary.LittleEndian.Uint16(data[local+28:]))
		if !fits(start, size) {
			return nil, fmt.Errorf("zip entry %s runs past the end of the file", name)
		}
		z.entries[name] = data[start : start+size]
	}
	return z, nil
}

// Pickle values produced by unpickle. Anything not listed decodes to Go
// nil, bool, int64, float64, string or []byte.
type (
	pickleGlobal struct{ module, name string }
	// pickleObject is an instance the pickle asked to construct: its class,
	// constructor arguments and, after BUILD, its __setstate__ payload
	// (for nn.Module, the instance __dict__).
	pickleObject struct {
		class pickleGlobal
		args  []any
		state any
	}
	pickleTuple []any
	pickleList  struct{ items []any }
	// pickleDict keeps insertion order, like the OrderedDicts nn.Module
	// uses; index makes lookups and duplicate-key updates O(1).
	pickleDict struct {
		keys, values []any
		index        map[any]int
	}
	// pickleStorage is a persistent-id reference to one data/<key> file.
	pickleStorage struct {
		dtype, key string
		numel      int
	}
	pickleTensor struct {
		storage      *pickleStorage
		offset       int
		size, stride []int
	}
	pickleMark struct{}
)

// pickleScalarKey reports whether k can index a pickleDict. nn.Module
// state only uses string and int keys; others (a tuple, say) are kept but
// never looked up, which also sidesteps Go's runtime panic for comparing
// uncomparable interface values.
func pickleScalarKey(k any) bool {
	switch k.(type) {
	case string, int64, bool:
		return true
	}
	return false
}

func (d *pickleDict) set(k, v any) {
	if pickleScalarKey(k) {
		if d.index == nil {
			d.index = map[any]int{}
		}
		if i, ok := d.index[k]; ok {
			d.values[i] = v
			return
		}
		d.index[k] = len(d.keys)
	}
	d.keys, d.values = append(d.keys, k), append(d.values, v)
}

func (d *pickleDict) get(k string) (any, bool) {
	if i, ok := d.index[k]; ok {
		return d.values[i], true
	}
	return nil, false
}

// unpickle interprets a pickle (protocols 2-5) without executing any of it.
func unpickle(data []byte) (any, error) {
	var stack []any
	memo := map[int]any{}
	pos := 0
	fail := func(format string, args ...any) error {
		return fmt.Errorf("pickle at byte %d: %s", pos, fmt.Sprintf(format, args...))
	}
	need := func(n int) ([]byte, error) {
		if n < 0 || pos+n > len(data) {
			return nil, fail("truncated")
		}
		b := data[pos : pos+n]
		pos += n
		return b, nil
	}
	pop := func() (any, error) {
		if len(stack) == 0 {
			return nil, fail("stack underflow")
		}
		v := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		return v, nil
	}
	popMark := func() ([]any, error) {
		for i := len(stack) - 1; i >= 0; i-- {
			if _, ok := stack[i].(pickleMark); ok {
				items := append([]any(nil), stack[i+1:]...)
				stack = stack[:i]
				return items, nil
			}
		}
		return nil, fail("missing MARK")
	}
	top := func() (any, error) {
		if len(stack) == 0 {
			return nil, fail("stack underflow")
		}
		return stack[len(stack)-1], nil
	}
	readLine := func() (string, error) {
		end := bytes.IndexByte(data[pos:], '\n')
		if end < 0 {
			return "", fail("unterminated line")
		}
		s := string(data[pos : pos+end])
		pos += end + 1
		return s, nil
	}
	lenPrefixed := func(width int) ([]byte, error) {
		b, err := need(width)
		if err != nil {
			return nil, err
		}
		var n uint64
		switch width {
		case 1:
			n = uint64(b[0])
		case 4:
			n = uint64(binary.LittleEndian.Uint32(b))
		case 8:
			n = binary.LittleEndian.Uint64(b)
		}
		if n > uint64(len(data)) {
			return nil, fail("length %d exceeds the pickle", n)
		}
		return need(int(n))
	}

	for {
		if pos >= len(data) {
			return nil, fail("missing STOP")
		}
		op := data[pos]
		pos++
		var push any
		hasPush := true
		switch op {
		case 0x80: // PROTO
			if _, err := need(1); err != nil {
				return nil, err
			}
			hasPush = false
		case 0x95: // FRAME
			if _, err := need(8); err != nil {
				return nil, err
			}
			hasPush = false
		case '.': // STOP
			return pop()
		case '(': // MARK
			push = pickleMark{}
		case '0': // POP
			if _, err := pop(); err != nil {
				return nil, err
			}
			hasPush = false
		case '1': // POP_MARK
			if _, err := popMark(); err != nil {
				return nil, err
			}
			hasPush = false
		case '2': // DUP
			v, err := top()
			if err != nil {
				return nil, err
			}
			push = v
		case 'N':
			push = nil
		case 0x88:
			push = true
		case 0x89:
			push = false
		case 'K': // BININT1
			b, err := need(1)
			if err != nil {
				return nil, err
			}
			push = int64(b[0])
		case 'M': // BININT2
			b, err := need(2)
			if err != nil {
				return nil, err
			}
			push = int64(binary.LittleEndian.Uint16(b))
		case 'J': // BININT
			b, err := need(4)
			if err != nil {
				return nil, err
			}
			push = int64(int32(binary.LittleEndian.Uint32(b)))
		case 0x8a, 0x8b: // LONG1, LONG4
			width := 1
			if op == 0x8b {
				width = 4
			}
			b, err := lenPrefixed(width)
			if err != nil {
				return nil, err
			}
			if len(b) > 8 {
				return nil, fail("integer wider than 64 bits")
			}
			var v int64
			for i := len(b) - 1; i >= 0; i-- {
				v = v<<8 | int64(b[i])
			}
			if len(b) > 0 && len(b) < 8 && b[len(b)-1]&0x80 != 0 {
				v -= 1 << (8 * len(b))
			}
			push = v
		case 'I': // INT (text; also "00"/"01" booleans)
			line, err := readLine()
			if err != nil {
				return nil, err
			}
			v, err := strconv.ParseInt(strings.TrimSpace(line), 10, 64)
			if err != nil {
				return nil, fail("bad INT %q", line)
			}
			push = v
		case 'G': // BINFLOAT (big-endian)
			b, err := need(8)
			if err != nil {
				return nil, err
			}
			push = math.Float64frombits(binary.BigEndian.Uint64(b))
		case 'X', 0x8c, 0x8d: // BINUNICODE, SHORT_BINUNICODE, BINUNICODE8
			width := map[byte]int{'X': 4, 0x8c: 1, 0x8d: 8}[op]
			b, err := lenPrefixed(width)
			if err != nil {
				return nil, err
			}
			push = string(b)
		case 'T', 'U': // BINSTRING, SHORT_BINSTRING (Python 2 str)
			width := 4
			if op == 'U' {
				width = 1
			}
			b, err := lenPrefixed(width)
			if err != nil {
				return nil, err
			}
			push = string(b)
		case 'B', 'C', 0x8e: // BINBYTES, SHORT_BINBYTES, BINBYTES8
			width := map[byte]int{'B': 4, 'C': 1, 0x8e: 8}[op]
			b, err := lenPrefixed(width)
			if err != nil {
				return nil, err
			}
			push = append([]byte(nil), b...)
		case ')':
			push = pickleTuple{}
		case 't':
			items, err := popMark()
			if err != nil {
				return nil, err
			}
			push = pickleTuple(items)
		case 0x85, 0x86, 0x87: // TUPLE1..3
			n := int(op - 0x84)
			if len(stack) < n {
				return nil, fail("stack underflow")
			}
			push = pickleTuple(append([]any(nil), stack[len(stack)-n:]...))
			stack = stack[:len(stack)-n]
		case ']':
			push = &pickleList{}
		case 'l':
			items, err := popMark()
			if err != nil {
				return nil, err
			}
			push = &pickleList{items: items}
		case 'a', 'e': // APPEND, APPENDS
			var items []any
			var err error
			if op == 'a' {
				var v any
				v, err = pop()
				items = []any{v}
			} else {
				items, err = popMark()
			}
			if err != nil {
				return nil, err
			}
			target, err := top()
			if err != nil {
				return nil, err
			}
			if l, ok := target.(*pickleList); ok {
				l.items = append(l.items, items...)
			}
			hasPush = false
		case '}':
			push = &pickleDict{}
		case 'd':
			items, err := popMark()
			if err != nil {
				return nil, err
			}
			if len(items)%2 != 0 {
				return nil, fail("odd DICT item count")
			}
			d := &pickleDict{}
			for i := 0; i < len(items); i += 2 {
				d.set(items[i], items[i+1])
			}
			push = d
		case 's', 'u': // SETITEM, SETITEMS
			var items []any
			if op == 's' {
				v, err := pop()
				if err != nil {
					return nil, err
				}
				k, err := pop()
				if err != nil {
					return nil, err
				}
				items = []any{k, v}
			} else {
				var err error
				if items, err = popMark(); err != nil {
					return nil, err
				}
			}
			if len(items)%2 != 0 {
				return nil, fail("odd SETITEMS item count")
			}
			target, err := top()
			if err != nil {
				return nil, err
			}
			if d, ok := target.(*pickleDict); ok {
				for i := 0; i < len(items); i += 2 {
					d.set(items[i], items[i+1])
				}
			}
			hasPush = false
		case 0x8f: // EMPTY_SET
			push = &pickleList{}
		case 0x90: // ADDITEMS
			items, err := popMark()
			if err != nil {
				return nil, err
			}
			target, err := top()
			if err != nil {
				return nil, err
			}
			if l, ok := target.(*pickleList); ok {
				l.items = append(l.items, items...)
			}
			hasPush = false
		case 0x91: // FROZENSET
			items, err := popMark()
			if err != nil {
				return nil, err
			}
			push = &pickleList{items: items}
		case 'c': // GLOBAL
			module, err := readLine()
			if err != nil {
				return nil, err
			}
			name, err := readLine()
			if err != nil {
				return nil, err
			}
			push = pickleGlobal{module, name}
		case 0x93: // STACK_GLOBAL
			name, err := pop()
			if err != nil {
				return nil, err
			}
			module, err := pop()
			if err != nil {
				return nil, err
			}
			ms, ok1 := module.(string)
			ns, ok2 := name.(string)
			if !ok1 || !ok2 {
				return nil, fail("STACK_GLOBAL needs two strings")
			}
			push = pickleGlobal{ms, ns}
		case 'R': // REDUCE
			args, err := pop()
			if err != nil {
				return nil, err
			}
			fn, err := pop()
			if err != nil {
				return nil, err
			}
			tuple, _ := args.(pickleTuple)
			push, err = pickleReduce(fn, tuple)
			if err != nil {
				return nil, fail("%v", err)
			}
		case 0x81, 0x92: // NEWOBJ, NEWOBJ_EX
			if op == 0x92 {
				if _, err := pop(); err != nil { // kwargs
					return nil, err
				}
			}
			args, err := pop()
			if err != nil {
				return nil, err
			}
			cls, err := pop()
			if err != nil {
				return nil, err
			}
			g, _ := cls.(pickleGlobal)
			tuple, _ := args.(pickleTuple)
			push = &pickleObject{class: g, args: tuple}
		case 'b': // BUILD
			state, err := pop()
			if err != nil {
				return nil, err
			}
			target, err := top()
			if err != nil {
				return nil, err
			}
			if o, ok := target.(*pickleObject); ok {
				o.state = state
			}
			hasPush = false
		case 'Q': // BINPERSID
			pid, err := pop()
			if err != nil {
				return nil, err
			}
			push, err = pickleStorageRef(pid)
			if err != nil {
				return nil, fail("%v", err)
			}
		case 'h', 'j', 'g': // BINGET, LONG_BINGET, GET
			var idx int
			switch op {
			case 'h':
				b, err := need(1)
				if err != nil {
					return nil, err
				}
				idx = int(b[0])
			case 'j':
				b, err := need(4)
				if err != nil {
					return nil, err
				}
				idx = int(binary.LittleEndian.Uint32(b))
			default:
				line, err := readLine()
				if err != nil {
					return nil, err
				}
				if idx, err = strconv.Atoi(line); err != nil {
					return nil, fail("bad GET %q", line)
				}
			}
			v, ok := memo[idx]
			if !ok {
				return nil, fail("memo %d is unset", idx)
			}
			push = v
		case 'q', 'r', 'p', 0x94: // BINPUT, LONG_BINPUT, PUT, MEMOIZE
			idx := len(memo)
			switch op {
			case 'q':
				b, err := need(1)
				if err != nil {
					return nil, err
				}
				idx = int(b[0])
			case 'r':
				b, err := need(4)
				if err != nil {
					return nil, err
				}
				idx = int(binary.LittleEndian.Uint32(b))
			case 'p':
				line, err := readLine()
				if err != nil {
					return nil, err
				}
				if idx, err = strconv.Atoi(line); err != nil {
					return nil, fail("bad PUT %q", line)
				}
			}
			v, err := top()
			if err != nil {
				return nil, err
			}
			memo[idx] = v
			hasPush = false
		default:
			return nil, fail("unsupported opcode 0x%02x", op)
		}
		if hasPush {
			stack = append(stack, push)
		}
	}
}

// pickleReduce reproduces the handful of pure data constructors torch.save
// output relies on and turns every other call into an inert pickleObject.
func pickleReduce(fn any, args pickleTuple) (any, error) {
	g, ok := fn.(pickleGlobal)
	if !ok {
		return &pickleObject{args: args}, nil
	}
	switch g.module + "." + g.name {
	case "torch._utils._rebuild_tensor", "torch._utils._rebuild_tensor_v2":
		if len(args) < 4 {
			return nil, fmt.Errorf("%s: %d arguments", g.name, len(args))
		}
		storage, ok := args[0].(*pickleStorage)
		offset, ok2 := args[1].(int64)
		size, ok3 := pickleInts(args[2])
		stride, ok4 := pickleInts(args[3])
		if !ok || !ok2 || !ok3 || !ok4 || len(size) != len(stride) {
			return nil, fmt.Errorf("%s: malformed arguments", g.name)
		}
		return &pickleTensor{storage: storage, offset: int(offset), size: size, stride: stride}, nil
	case "torch._utils._rebuild_parameter", "torch._utils._rebuild_parameter_with_state":
		if len(args) == 0 {
			return nil, fmt.Errorf("%s: no tensor", g.name)
		}
		return args[0], nil
	case "collections.OrderedDict":
		d := &pickleDict{}
		if len(args) > 0 {
			if l, ok := args[0].(*pickleList); ok {
				for _, item := range l.items {
					if kv, ok := item.(pickleTuple); ok && len(kv) == 2 {
						d.set(kv[0], kv[1])
					}
				}
			}
		}
		return d, nil
	case "__builtin__.set", "builtins.set", "__builtin__.frozenset", "builtins.frozenset":
		if len(args) > 0 {
			if l, ok := args[0].(*pickleList); ok {
				return &pickleList{items: append([]any(nil), l.items...)}, nil
			}
		}
		return &pickleList{}, nil
	case "torch.Size":
		if len(args) > 0 {
			return args[0], nil
		}
		return pickleTuple{}, nil
	}
	return &pickleObject{class: g, args: args}, nil
}

func pickleInts(v any) ([]int, bool) {
	t, ok := v.(pickleTuple)
	if !ok {
		return nil, false
	}
	out := make([]int, len(t))
	for i, x := range t {
		n, ok := x.(int64)
		if !ok || n < 0 {
			return nil, false
		}
		out[i] = int(n)
	}
	return out, true
}

// pickleStorageRef decodes torch.save's persistent id,
// ('storage', <StorageType>, key, location, numel).
func pickleStorageRef(pid any) (any, error) {
	t, ok := pid.(pickleTuple)
	if !ok || len(t) < 5 || t[0] != "storage" {
		return nil, fmt.Errorf("unsupported persistent id %v", pid)
	}
	g, ok1 := t[1].(pickleGlobal)
	key, ok2 := t[2].(string)
	numel, ok3 := t[4].(int64)
	if !ok1 || !ok2 || !ok3 {
		return nil, fmt.Errorf("malformed storage reference %v", pid)
	}
	return &pickleStorage{dtype: g.name, key: key, numel: int(numel)}, nil
}

// torchCheckpoint is an opened torch.save archive: its unpickled root
// object and the raw storages its tensors point into.
type torchCheckpoint struct {
	root     any
	storages map[string][]byte
}

func openTorchCheckpoint(data []byte) (*torchCheckpoint, error) {
	z, err := openTorchZip(data)
	if err != nil {
		return nil, fmt.Errorf("reading PyTorch checkpoint: %w", err)
	}
	prefix := ""
	var pkl []byte
	for name, entry := range z.entries {
		if strings.HasSuffix(name, "/data.pkl") && strings.Count(name, "/") == 1 {
			prefix, pkl = strings.TrimSuffix(name, "data.pkl"), entry
			break
		}
	}
	if pkl == nil {
		return nil, errors.New("reading PyTorch checkpoint: no data.pkl (legacy non-zip torch.save files are not supported)")
	}
	if order, ok := z.entries[prefix+"byteorder"]; ok && strings.TrimSpace(string(order)) != "little" {
		return nil, fmt.Errorf("reading PyTorch checkpoint: byte order %q is not supported", order)
	}
	root, err := unpickle(pkl)
	if err != nil {
		return nil, fmt.Errorf("reading PyTorch checkpoint: %w", err)
	}
	storages := map[string][]byte{}
	for name, entry := range z.entries {
		if key, ok := strings.CutPrefix(name, prefix+"data/"); ok {
			storages[key] = entry
		}
	}
	return &torchCheckpoint{root: root, storages: storages}, nil
}

// moduleStateDict flattens an unpickled nn.Module tree into state_dict
// names ("model.0.conv.weight") and tensors, the way Module.state_dict()
// would, by walking each instance's _parameters, _buffers and _modules.
func moduleStateDict(module any, prefix string, out map[string]*pickleTensor) {
	obj, ok := module.(*pickleObject)
	if !ok {
		return
	}
	state, ok := obj.state.(*pickleDict)
	if !ok {
		return
	}
	for _, field := range []string{"_parameters", "_buffers"} {
		if d, ok := state.get(field); ok {
			if d, ok := d.(*pickleDict); ok {
				for i, k := range d.keys {
					name, _ := k.(string)
					if t, ok := d.values[i].(*pickleTensor); ok && name != "" {
						out[prefix+name] = t
					}
				}
			}
		}
	}
	if d, ok := state.get("_modules"); ok {
		if d, ok := d.(*pickleDict); ok {
			for i, k := range d.keys {
				if name, ok := k.(string); ok {
					moduleStateDict(d.values[i], prefix+name+".", out)
				}
			}
		}
	}
}

// torchElementSize maps a storage class to its element width and whether
// tensorF32 can decode it.
func torchElementSize(dtype string) (int, bool) {
	switch dtype {
	case "FloatStorage":
		return 4, true
	case "HalfStorage", "BFloat16Storage":
		return 2, true
	case "DoubleStorage":
		return 8, true
	case "LongStorage":
		return 8, false
	case "IntStorage":
		return 4, false
	case "ShortStorage":
		return 2, false
	case "CharStorage", "ByteStorage", "BoolStorage":
		return 1, false
	}
	return 0, false
}

// tensorF32 materialises a floating-point tensor as contiguous row-major
// float32, honouring its storage offset and strides.
func (c *torchCheckpoint) tensorF32(t *pickleTensor) ([]float32, error) {
	width, ok := torchElementSize(t.storage.dtype)
	if !ok {
		return nil, fmt.Errorf("tensor storage %s is not floating point", t.storage.dtype)
	}
	raw, ok := c.storages[t.storage.key]
	if !ok {
		return nil, fmt.Errorf("tensor storage data/%s is missing", t.storage.key)
	}
	elements := len(raw) / width
	// Weights never broadcast, so a tensor cannot hold more elements than
	// its storage; checking as the product grows also rules out overflow
	// and absurd allocations from a malformed size tuple.
	numel := 1
	for _, d := range t.size {
		if d != 0 && numel > elements/d {
			return nil, fmt.Errorf("tensor shape %v exceeds its %d-element storage data/%s", t.size, elements, t.storage.key)
		}
		numel *= d
	}
	out := make([]float32, numel)
	idx := make([]int, len(t.size))
	for i := range out {
		at := t.offset
		for d, v := range idx {
			at += v * t.stride[d]
		}
		if at < 0 || at >= elements {
			return nil, fmt.Errorf("tensor element %d lies outside storage data/%s", at, t.storage.key)
		}
		b := raw[at*width:]
		switch t.storage.dtype {
		case "FloatStorage":
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b))
		case "HalfStorage":
			out[i] = numeric.F16ToF32(binary.LittleEndian.Uint16(b))
		case "BFloat16Storage":
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b)) << 16)
		case "DoubleStorage":
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(b)))
		}
		for d := len(idx) - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < t.size[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out, nil
}
