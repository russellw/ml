'use strict'
const assert = require('assert')
const etc = require('./etc')
const cnf = require('./cnf')

function simplify(clauses, m) {
	var cs = clauses.map((c) => cnf.simplify(c, m))
	return cs.filter((c) => !etc.eq(c, [[], [true]]))
}

function unit(cs) {
	for (var c of cs) {
		var [neg, pos] = c
		if (neg.length === 1 && pos.length === 0) return etc.mk('unit', neg[0], false)
		if (neg.length === 0 && pos.length === 1) return etc.mk('unit', pos[0], true)
	}
}

function sat(clauses, deadline) {
	var ps = []
	loop: for (;;) {
		etc.cktime(deadline)

		var m = new Map(ps)
		var cs = simplify(clauses, m)

		// unit propagation
		var p = unit(cs)
		if (p) {
			ps.push(p)
			continue
		}

		// solution
		if (!cs.length) return m

		// contradiction: backtrack
		for (var c of cs)
			if (etc.eq(c, [[], []]))
				for (var i = ps.length; ; i--) {
					if (!i) return
					var p = ps[i - 1]
					if (p.o === 'guess' && !p[1]) {
						p[1] = true
						continue loop
					}
				}

		// unassigned atoms
		var atoms = new Set()
		for (var c of cs) for (var L of c) for (var a of L) atoms.add(a)
		assert(atoms.size)

		// pick one and guess
		var a = [...atoms][0]
		ps.push(etc.mk('guess', a, false))
	}
}

function solve(clauses, deadline) {
	var solution = sat(clauses, deadline)
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
