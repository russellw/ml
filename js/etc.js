'use strict'
const assert = require('assert')

function eq(a, b) {
	if (a === b) return true
	if (!Array.isArray(a)) return
	if (a.o !== b.o) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length; i++) if (!eq(a[i], b[i])) return
	return true
}

function map(a, f) {
	if (!Array.isArray(a)) return a
	var r = []
	Object.assign(r, a)
	for (var i = 0; i < r.length; i++) r[i] = f(r[i])
	return r
}

function cartproduct(qs) {
	var js = []
	for (var q of qs) js.push(0)
	var rs = []

	function rec(i) {
		if (i === js.length) {
			var ys = []
			for (i = 0; i < js.length; i++) ys.push(qs[i][js[i]])
			rs.push(ys)
			return
		}
		for (js[i] = 0; js[i] < qs[i].length; js[i]++) rec(i + 1)
	}

	rec(0)
	return rs
}

function mk(o, ...args) {
	var a = Array.from(args)
	a.o = o
	return a
}

function replace(a, m) {
	if (m.has(a)) return replace(m.get(a), m)
	return map(a, (b) => replace(b, m))
}

function getor(m, k, f) {
	if (m.has(k)) return m.get(k)
	var v = f()
	m.set(k, v)
	return v
}

function walk(a, f) {
	f(a)
	if (Array.isArray(a)) for (var b of a) walk(b, f)
}

function err(file, text, toki, msg) {
	// line number
	var line = 1
	for (var i = 0; i < toki; i++) if (text[i] === '\n') line++

	// start of line
	var linestart = toki
	while (linestart && text[linestart - 1] !== '\n') linestart--

	// print context
	for (var i = linestart; text[i] >= ' ' || text[i] === '\t'; i++);
	console.error(text.slice(linestart, i))

	// print caret
	for (var i = linestart; i < toki; i++) process.stderr.write(text[i] === '\t' ? '\t' : ' ')
	console.error('^')

	// print message and exit
	console.error('%s:%d: %s', file, line, msg)
	process.exit(1)
}

// default param
function f(a = []) {
	a.push(1)
	return a
}

assert(f().length === 1)
assert(f().length === 1)

// getor
var m = new Map()

assert(getor(m, 'a', () => 5) === 5)
assert(m.size === 1)
assert(m.get('a') === 5)

assert(getor(m, 'a', () => 5) === 5)
assert(m.size === 1)
assert(m.get('a') === 5)

assert(getor(m, 'b', () => 6) === 6)
assert(m.size === 2)
assert(m.get('a') === 5)
assert(m.get('b') === 6)

// concat
function g(...s) {
	return s
}

assert(g(1, 2).length === 2)
assert(g(...[1, 2]).length === 2)
assert(g(...[1, 2].concat([3, 4])).length === 4)

// cartesian product
var qs = []
var q
q = []
q.push('a0')
q.push('a1')
qs.push(q)
q = []
q.push('b0')
q.push('b1')
q.push('b2')
qs.push(q)
q = []
q.push('c0')
q.push('c1')
q.push('c2')
q.push('c3')
qs.push(q)
var rs = cartproduct(qs)
var i = 0
assert(eq(rs[i++], ['a0', 'b0', 'c0']))
assert(eq(rs[i++], ['a0', 'b0', 'c1']))
assert(eq(rs[i++], ['a0', 'b0', 'c2']))
assert(eq(rs[i++], ['a0', 'b0', 'c3']))
assert(eq(rs[i++], ['a0', 'b1', 'c0']))
assert(eq(rs[i++], ['a0', 'b1', 'c1']))
assert(eq(rs[i++], ['a0', 'b1', 'c2']))
assert(eq(rs[i++], ['a0', 'b1', 'c3']))
assert(eq(rs[i++], ['a0', 'b2', 'c0']))
assert(eq(rs[i++], ['a0', 'b2', 'c1']))
assert(eq(rs[i++], ['a0', 'b2', 'c2']))
assert(eq(rs[i++], ['a0', 'b2', 'c3']))
assert(eq(rs[i++], ['a1', 'b0', 'c0']))
assert(eq(rs[i++], ['a1', 'b0', 'c1']))
assert(eq(rs[i++], ['a1', 'b0', 'c2']))
assert(eq(rs[i++], ['a1', 'b0', 'c3']))
assert(eq(rs[i++], ['a1', 'b1', 'c0']))
assert(eq(rs[i++], ['a1', 'b1', 'c1']))
assert(eq(rs[i++], ['a1', 'b1', 'c2']))
assert(eq(rs[i++], ['a1', 'b1', 'c3']))
assert(eq(rs[i++], ['a1', 'b2', 'c0']))
assert(eq(rs[i++], ['a1', 'b2', 'c1']))
assert(eq(rs[i++], ['a1', 'b2', 'c2']))
assert(eq(rs[i++], ['a1', 'b2', 'c3']))

// map
assert(
	!eq(
		mk('+', 1, 2).map((a) => a + 10),
		mk('+', 11, 12)
	)
)
assert(
	eq(
		map(mk('+', 1, 2), (a) => a + 10),
		mk('+', 11, 12)
	)
)

// exports
exports.err = err
exports.walk = walk
exports.getor = getor
exports.eq = eq
exports.map = map
exports.mk = mk
exports.replace = replace
