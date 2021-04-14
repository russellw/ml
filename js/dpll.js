'use strict'
const assert = require('assert')
const logic = require('./logic')
const etc = require('./etc')
const cnf = require('./cnf')

function sat(clauses, m = new Map()) {
	var cs = clauses.map((c) => cnf.clause(c[0], c[1], m))
	for (var c in cs) if (logic.eq(c, cnf.falseClause)) return
	cs = cs.filter((c) => !logic.eq(c, cnf.trueClause))
	if (!cs.length) return m

	// unit clauses
	for (var c of cs) {
		var [neg, pos] = c
		if (neg.length + pos.length === 1) {
			if (neg.length) m.set(neg[0], false)
			else m.set(pos[0], true)
			return sat(clauses, m)
		}
	}

	// atoms
	var atoms = new Set()
	etc.walk(cs, (a) => {
		if (a.op === 'fn') atoms.add(a)
	})

	// pure atoms
	function occurs(pol, a) {
		for (var c of cs) if (c[pol].includes(a)) return true
	}

	for (var a of atoms)
		if (occurs(0, a) !== occurs(1, a)) {
			m.set(a, !!occurs(1, a))
			return sat(clauses, m)
		}

	// guess
	for (var a of atoms) {
		var m1 = new map(m)
		m1.set(a, false)
		var r = sat(clauses, m1)
		if (r) return r
		m.set(a, true)
		return sat(clauses, m)
	}
}

var m = sat([[[], []]])
assert(!m)

var a = { op: 'fn' }
m = sat([[[], [a]]])
assert(m)
assert(m.size === 1)
assert(m.get(a) === true)

var b = { op: 'fn' }
m = sat([
	[[], [a]],
	[[], [b]],
])
assert(m)
assert(m.size === 2)
assert(m.get(a) === true)
assert(m.get(b) === true)

m = sat([
	[[a], []],
	[[b], []],
])
assert(m)
assert(m.size === 2)
assert(m.get(a) === false)
assert(m.get(b) === false)

m = sat([
	[[a], []],
	[[], [a]],
])
assert(!m)

exports.sat = sat
