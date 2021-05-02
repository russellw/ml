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
			if (etc.eq(c, [[], []])) {
				while (ps.length) {
					var p = ps.pop()
					if (p.o === 'guess' && !p[1]) {
						ps.push(etc.mk('guess', p[0], true))
						continue loop
					}
				}
				return
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

	function dpll(clauses, m, deadline) {
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
				return dpll(clauses, m, deadline)
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
				return dpll(clauses, m, deadline)
			}

		// guess
		for (var a of atoms) {
			etc.cktime(deadline)
			var m1 = new Map(m)
			m1.set(a, false)
			var r = dpll(clauses, m1, deadline)
			if (r) return r
			m.set(a, true)
			return dpll(clauses, m, deadline)
		}
	}

	var m = dpll([[[], []]], new Map())
	assert(!m)

	var a = {}
	m = dpll([[[], [a]]], new Map())
	assert(m)
	assert(m.size === 1)
	assert(m.get(a) === true)

	var b = {}
	m = dpll(
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

	m = dpll(
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

	m = dpll(
		[
			[[a], []],
			[[], [a]],
		],
		new Map()
	)
	assert(!m)

	var atoms = []
	for (var i = 0; i < 10; i++) atoms.push({ o: 'fn', name: i, type: 'boolean' })

	for (var i = 0; i < 10; i++) {
		var clauses = []
		var nc = Math.random() * 10
		for (var j = 0; j < nc; j++) {
			var c = [[], []]
			var nL = Math.random() * 10
			for (var k = 0; k < nL; k++) {
				var a = atoms[Math.floor(Math.random() * atoms.length)]
				c[Math.floor(Math.random() * 2)].push(a)
			}
			clauses.push(c)
		}
		var s1 = !!dpll(clauses, new Map())
		var s2 = !!sat(clauses)
		if (s1 !== s2) etc.show(clauses)
		assert(s1 === s2)
	}
}

test()

module.exports = {
	solve,
}
