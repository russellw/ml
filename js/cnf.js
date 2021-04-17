'use strict'
const logic = require('./logic')
const etc = require('./etc')
const assert = require('assert')

function clause(neg, pos, m = new Map()) {
	// simplify
	neg = neg.map((a) => logic.simplify(a, m))
	pos = pos.map((a) => logic.simplify(a, m))

	// filter out redundancy
	neg = neg.filter((a) => a !== true)
	pos = pos.filter((a) => a !== false)

	// tautology?
	for (var a of neg) if (a === false) return [[], [true]]
	for (var a of pos) if (a === true) return [[], [true]]
	for (var a of neg) for (var b of pos) if (etc.eq(a, b)) return [[], [true]]

	// make new clause
	return [neg, pos]
}

function convert(c, clauses) {
	function all(bound, pol, a) {
		bound = new Map(bound)
		for (var x of a[0]) bound.set(x, { o: 'var' })
		return nnf(bound, pol, a[1])
	}

	function exists(bound, pol, a) {
		var free = logic.freevars(a[1])
		var params = []
		for (var [k, v] of bound.entries()) if (v.o === 'var' && free.has(k)) params.push(v)
		bound = new Map(bound)
		for (var x of a[0]) {
			var sk = {}
			if (params.length) sk = etc.mk('call', ...[sk].concat(params))
			bound.set(x, sk)
		}
		return nnf(bound, pol, a[1])
	}

	// most of the work is done in conversion to negation normal form
	function nnf(bound, pol, a) {
		switch (a) {
			case false:
				return !pol
			case true:
				return pol
		}
		switch (a.o) {
			case 'all':
				return (pol ? all : exists)(bound, pol, a)
			case 'exists':
				return (pol ? exists : all)(bound, pol, a)
			case '!':
				return nnf(bound, !pol, a[0])
			case '=>':
				return nnf(bound, pol, etc.mk('||', etc.mk('!', a[0]), a[1]))
			case '&&':
				return etc.mk(pol ? '&&' : '||', ...a.map((b) => nnf(bound, pol, b)))
			case '||':
				return etc.mk(pol ? '||' : '&&', ...a.map((b) => nnf(bound, pol, b)))
			case 'var':
				assert(bound.has(a))
				return bound.get(a)
			case '<=>':
				var x = a[0]
				var y = a[1]
				return nnf(bound, pol, etc.mk('&&', etc.mk('=>', x, y), etc.mk('=>', y, x)))
		}
		a = etc.map(a, (b) => nnf(bound, pol, b))
		return pol ? a : etc.mk('!', a)
	}

	// make AND rise to the top
	function rise(a) {
		a = etc.map(a, rise)
		if (a.o !== '||') return a

		// now we know this term is an OR
		// its arguments can be taken without loss of generality as ANDs
		var ands = []
		for (var b of a) {
			var and = []
			flatten('&&', b, and)
			ands.push(and)
		}

		// OR distributes over AND by Cartesian product
		a = etc.cartproduct(ands)
		for (var b of a) b.o = '||'
		a.o = '&&'
		return a
	}

	var a = c[0]
	a = nnf(new Map(), true, a)
	a = rise(a)

	// now we have a term in CNF
	// need to convert it to actual clauses
	var ors = []
	flatten('&&', a, ors)
	for (var b of ors) {
		var d = cterm(b)
		if (etc.eq(d, [[], [true]])) continue
		d.from = [c]
		clauses.push(d)
	}
}

function flatten(o, a, r) {
	if (a.o === o) {
		for (var b of a) flatten(o, b, r)
		return
	}
	r.push(a)
}

function cterm(a) {
	var neg = []
	var pos = []

	function rec(a) {
		assert(a.o !== '&&')
		switch (a.o) {
			case '||':
				for (var b of a) rec(b)
				return
			case '!':
				neg.push(a[0])
				return
		}
		pos.push(a)
	}

	rec(a)
	return clause(neg, pos)
}

// clause
var a = {}
var b = {}
assert(etc.eq(clause([a], [b]), clause([a], [b])))
assert(etc.eq(clause([a], [b]), [[a], [b]]))
assert(etc.eq(clause([a], [b]), cterm(etc.mk('||', etc.mk('!', a), b))))
assert(etc.eq(clause([a], [false]), [[a], []]))
assert(etc.eq(clause([a], [true]), [[], [true]]))
assert(etc.eq(clause([a], [a]), [[], [true]]))
assert(etc.eq([[], [true]], [[], [true]]))
assert(!etc.eq([[], [true]], [[], []]))
assert(etc.eq([[], []], [[], []]))

// flatten
var r = []
flatten('+', etc.mk('+', etc.mk('+', 1, 2), 3), r)
assert(etc.eq(r, [1, 2, 3]))

var r = []
flatten('+', 4, r)
assert(etc.eq(r, [4]))

// convert
var cs = []
convert([true], cs)
assert(cs.length === 0)

var cs = []
convert([false], cs)
assert(cs.length === 1)
assert(etc.eq(cs[0], [[], []]))

var cs = []
convert([a], cs)
assert(cs.length === 1)
assert(etc.eq(cs[0], [[], [a]]))

var cs = []
convert([etc.mk('!', a)], cs)
assert(cs.length === 1)
assert(etc.eq(cs[0], [[a], []]))

var cs = []
convert([etc.mk('!', etc.mk('!', a))], cs)
assert(cs.length === 1)
assert(etc.eq(cs[0], [[], [a]]))

var cs = []
convert([etc.mk('=>', a, b)], cs)
assert(cs.length === 1)
assert(etc.eq(cs[0], [[a], [b]]))

var cs = []
convert([etc.mk('||', a, b)], cs)
assert(cs.length === 1)
assert(etc.eq(cs[0], [[], [a, b]]))

var cs = []
convert([etc.mk('&&', a, b)], cs)
assert(cs.length === 2)
assert(etc.eq(cs[0], [[], [a]]))
assert(etc.eq(cs[1], [[], [b]]))

var a1 = {}
var b1 = {}
var a2 = {}
var b2 = {}

var cs = []
convert([etc.mk('||', a, b, a1, b1)], cs)
assert(cs.length === 1)
assert(etc.eq(cs[0], [[], [a, b, a1, b1]]))

var cs = []
convert([etc.mk('||', etc.mk('||', a1, b1), etc.mk('||', a2, b2))], cs)
assert(cs.length === 1)
assert(etc.eq(cs[0], [[], [a1, b1, a2, b2]]))

var cs = []
convert([etc.mk('&&', etc.mk('||', a1, b1), etc.mk('||', a2, b2))], cs)
assert(cs.length === 2)
assert(etc.eq(cs[0], [[], [a1, b1]]))
assert(etc.eq(cs[1], [[], [a2, b2]]))

var cs = []
convert([etc.mk('||', a, etc.mk('&&', b1, b2))], cs)
assert(cs.length === 2)
assert(etc.eq(cs[0], [[], [a, b1]]))
assert(etc.eq(cs[1], [[], [a, b2]]))

var x = { o: 'var' }
var y = { o: 'var' }
var z = { o: 'var' }

assert(logic.match(etc.mk('call', a, x), etc.mk('call', a, 1)))
assert(!logic.match(etc.mk('call', a, 1), etc.mk('call', a, x)))

function isomorphic(a, b) {
	return logic.match(a, b) && logic.match(b, a)
}

var cs = []
convert([etc.mk('all', [x], etc.mk('call', a, x))], cs)
assert(cs.length === 1)
assert(isomorphic(cs[0], [[], [etc.mk('call', a, x)]]))

var cs = []
convert([etc.mk('all', [x, y], etc.mk('call', a, x, y))], cs)
assert(cs.length === 1)
assert(isomorphic(cs[0], [[], [etc.mk('call', a, x, y)]]))

var cs = []
convert([etc.mk('all', [x], etc.mk('all', [y], etc.mk('call', a, x, y)))], cs)
assert(cs.length === 1)
assert(isomorphic(cs[0], [[], [etc.mk('call', a, x, y)]]))

var cs = []
convert([etc.mk('exists', [x], etc.mk('call', a, x))], cs)
assert(cs.length === 1)
var m = logic.match([etc.mk('call', a, x)], cs[0][1])
assert(m)
assert(m.size === 1)
assert(!Array.isArray(m.get(x)))
assert(!m.get(x).o)

var cs = []
convert([etc.mk('all', [x], etc.mk('exists', [y], etc.mk('call', a, x, y)))], cs)
assert(cs.length === 1)
var m = logic.match([etc.mk('call', a, x, y)], cs[0][1])
assert(m)
assert(m.size === 2)
assert(!Array.isArray(m.get(x)))
assert(m.get(x).o === 'var')
assert(Array.isArray(m.get(y)))
assert(m.get(y).o === 'call')
assert(m.get(y).length === 2)

var cs = []
convert([etc.mk('all', [x], etc.mk('exists', [y], etc.mk('call', a, y)))], cs)
assert(cs.length === 1)
var m = logic.match([etc.mk('call', a, y)], cs[0][1])
assert(m)
assert(m.size === 1)
assert(!Array.isArray(m.get(y)))
assert(!m.get(y).o)

var cs = []
convert([etc.mk('<=>', a, b)], cs)
assert(cs.length === 2)
assert(etc.eq(cs[0], [[a], [b]]))
assert(etc.eq(cs[1], [[b], [a]]))

function sat(clauses) {
	var atoms = new Set()
	for (var c of cs) for (var p of c) for (var a of p) atoms.add(a)
	atoms = Array.from(atoms)

	function rec(i, m) {
		var cs = clauses.map((c) => clause(c[0], c[1], m))
		for (var c of cs) if (etc.eq(c, [[], []])) return
		cs = cs.filter((c) => !etc.eq(c, [[], [true]]))
		if (!cs.length) return m

		assert(i < atoms.length)
		var a = atoms[i++]
		var m1 = new Map(m)
		m1.set(a, false)
		if (rec(i, m1)) return true
		m.set(a, true)
		return rec(i, m)
	}

	return rec(0, new Map())
}

var cs = []
convert([etc.mk('&&', a, b)], cs)
assert(sat(cs))

var cs = []
convert([etc.mk('&&', a, a)], cs)
assert(sat(cs))

var cs = []
convert([etc.mk('&&', a, etc.mk('!', a))], cs)
assert(!sat(cs))

function thm(a) {
	var cs = []
	convert([a], cs)
	assert(sat(cs))

	var cs = []
	convert([etc.mk('!', a)], cs)
	assert(!sat(cs))
}

thm(true)
thm(etc.mk('=>', false, a))
thm(etc.mk('=>', a, a))
thm(etc.mk('&&', true, true, true))
thm(etc.mk('||', false, false, true))
thm(etc.mk('<=>', a, a))

var p1 = {}
var p2 = {}
var p3 = {}

thm(etc.mk('<=>', p1, etc.mk('<=>', p2, etc.mk('<=>', p1, p2))))
thm(etc.mk('<=>', p1, etc.mk('<=>', p2, etc.mk('<=>', p3, etc.mk('<=>', p1, etc.mk('<=>', p2, p3))))))

// exports
exports.clause = clause
exports.convert = convert
