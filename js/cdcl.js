'use strict'
const assert = require('assert')
const etc = require('./etc')
const cnf = require('./cnf')

function simplify(clauses, m) {
	var cs = clauses.map((c) => cnf.simplify(c, m))
	return cs.filter((c) => !etc.eq(c, [[], [true]]))
}

function unit(clauses, m) {
	var cs = simplify(cs, m)
	for (var c of cs) {
		var [neg, pos] = c
		if (neg.length === 1 && pos.length === 0) return [neg[0], false]
		if (neg.length === 0 && pos.length === 1) return [pos[0], true]
	}
}

function sat(clauses, deadline) {
	// atoms
	var atoms = new Set()
	for (var c of cs) for (var L of c) for (var a of L) atoms.add(a)

	var ps = []
	for (;;) {
		for (;;) {
			var p = unit(clauses, new Map(ps))
			if (!p) break
			ps.push(p)
		}
		for (var c of cs) if (etc.eq(c, [[], []])) return
		var i = vs.length
		var a = atoms[i]
		vs.push(false)
		var m = mkmap(atoms, vs)
		var cs = simplify(cs, m)
	}
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
