'use strict'
const assert = require('assert')
const etc = require('./etc')
const cnf = require('./cnf')

function sat(clauses, m = new Map()) {
	var cs = clauses.map((c) => cnf.simplify(c, m))
	for (var c of cs) if (etc.eq(c, [[], []])) return
	cs = cs.filter((c) => !etc.eq(c, [[], [true]]))
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
	for (var c of cs) for (var L of c) for (var a of L) atoms.add(a)

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
		var m1 = new Map(m)
		m1.set(a, false)
		var r = sat(clauses, m1)
		if (r) return r
		m.set(a, true)
		return sat(clauses, m)
	}
}

function solve(clauses) {
	var solution = sat(clauses)
	return { szs: solution ? 'Satisfiable' : 'Unsatisfiable', solution }
}

function test() {
	var m = sat([[[], []]])
	assert(!m)

	var a = {}
	m = sat([[[], [a]]])
	assert(m)
	assert(m.size === 1)
	assert(m.get(a) === true)

	var b = {}
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
}

test()

exports.solve = solve
