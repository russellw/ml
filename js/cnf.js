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
	for (var a of neg) for (var b of pos) if (logic.eq(a, b)) return [[], [true]]

	// make new clause
	return [neg, pos]
}

function convert(c, clauses) {
	assert(c.o === 'fof')

	function all(bound, pol, a) {
		bound = new Map(bound)
		for (var x of a[0]) bound.set(x, { o: 'var' })
		return nnf(bound, pol, a[1])
	}

	function exists(bound, pol, a) {
		var params = []
		for (var x of logic.freevars(a[1])) {
			assert(bound.has(x))
			if (bound.get(x).o === 'var') params.push(bound.get(x))
		}
		bound = new Map(bound)
		for (var x of a[0]) {
			var sk = {}
			if (params.length) sk = logic.term('call', ...[sk].concat(params))
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
				return nnf(bound, !pol, a)
			case '=>':
				return nnf(bound, pol, logic.term('||', logic.term('!', a[0]), a[1]))
			case '&&':
				return logic.term(pol ? '&&' : '||', ...(b) => nnf(bound, pol, b))
			case '||':
				return logic.term(pol ? '||' : '&&', ...(b) => nnf(bound, pol, b))
		}
	}

	// make AND rise to the top
	function rise(a) {
		a = logic.map(a, rise)
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
		if (etc.eq(d, [[][true]])) continue
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
assert(logic.eq(clause([a], [b]), clause([a], [b])))
assert(logic.eq(clause([a], [b]), [[a], [b]]))
assert(logic.eq(clause([a], [b]), cterm(logic.term('||', logic.term('!', a), b))))
assert(logic.eq(clause([a], [false]), [[a], []]))
assert(logic.eq(clause([a], [true]), [[], [true]]))
assert(logic.eq(clause([a], [a]), [[], [true]]))
assert(logic.eq([[], [true]], [[], [true]]))
assert(!logic.eq([[], [true]], [[], []]))
assert(logic.eq([[], []], [[], []]))

// flatten
var r = []
flatten('+', logic.term('+', logic.term('+', 1, 2), 3), r)
assert(etc.eq(r, [1, 2, 3]))

var r = []
flatten('+', 4, r)
assert(etc.eq(r, [4]))

// convert
var cs = []
var c = logic.term('fof', true)
convert(c, cs)
assert(cs.length === 0)

// exports
exports.clause = clause
