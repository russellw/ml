'use strict'
const assert = require('assert')
const etc = require('./etc')
const cnf = require('./cnf')

function sat(clauses, m, deadline) {
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
			return sat(clauses, m, deadline)
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
			return sat(clauses, m, deadline)
		}

	// guess
	for (var a of atoms) {
		etc.cktime(deadline)
		var m1 = new Map(m)
		m1.set(a, false)
		var r = sat(clauses, m1, deadline)
		if (r) return r
		m.set(a, true)
		return sat(clauses, m, deadline)
	}
}

function solve(clauses, deadline) {
	var solution = sat(clauses, new Map(), deadline)
	return { szs: solution ? 'Satisfiable' : 'Unsatisfiable', solution }
}

function test() {
	var m = sat([[[], []]], new Map())
	assert(!m)

	var a = {}
	m = sat([[[], [a]]], new Map())
	assert(m)
	assert(m.size === 1)
	assert(m.get(a) === true)

	var b = {}
	m = sat(
		[
			[[], [a]],
			[[], [b]],
		],
		new Map()
	)
	assert(m)
	assert(m.size === 2)
	assert(m.get(a) === true)
	assert(m.get(b) === true)

	m = sat(
		[
			[[a], []],
			[[b], []],
		],
		new Map()
	)
	assert(m)
	assert(m.size === 2)
	assert(m.get(a) === false)
	assert(m.get(b) === false)

	m = sat(
		[
			[[a], []],
			[[], [a]],
		],
		new Map()
	)
	assert(!m)
}

test()

exports.solve = solve
