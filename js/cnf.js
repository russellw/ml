'use strict'
const logic = require('./logic')
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
	assert(c.op === 'fof')

	function nnf(all, exists, pol, a) {
		switch (a) {
			case false:
				return !pol
			case true:
				return pol
		}
		switch (a.op) {
			case '!':
				return nnf(all, exists, !pol, a)
			case '=>':
				return nnf(all, exists, pol, logic.term('||', logic.term('!', a[0]), a[1]))
		}
	}

	var a = c[0]
	a = nnf(new Map(), new Map(), true, a)
}

function clauseTerm(a) {
	var neg = []
	var pos = []

	function rec(a) {
		assert(a.op !== '&&')
		switch (a.op) {
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

var a = {}
var b = {}
assert(logic.eq(clause([a], [b]), clause([a], [b])))
assert(logic.eq(clause([a], [b]), [[a], [b]]))
assert(logic.eq(clause([a], [b]), clauseTerm(logic.term('||', logic.term('!', a), b))))
assert(logic.eq(clause([a], [false]), [[a], []]))
assert(logic.eq(clause([a], [true]), [[], [true]]))
assert(logic.eq(clause([a], [a]), [[], [true]]))
assert(logic.eq([[], [true]], [[], [true]]))
assert(!logic.eq([[], [true]], [[], []]))
assert(logic.eq([[], []], [[], []]))

exports.clause = clause
