'use strict'
const cdcl = require('./cdcl')
const etc = require('./etc')
const cnf = require('./cnf')
const superposition = require('./superposition')
const assert = require('assert')

function propositional(clauses) {
	for (var c of clauses) for (var L of c) for (var a of L) if (Array.isArray(a)) return
	return true
}

function solve(problem, deadline) {
	for (var c of problem.formulas) cnf.convert(c, problem.clauses)
	if (propositional(problem.clauses)) return cdcl.solve(problem.clauses, deadline)
	return superposition.solve(problem.clauses, deadline)
}

function test() {
	function sat(cs) {
		var r1 = superposition.solve(cs).szs
		if (propositional(cs)) {
			var r2 = cdcl.solve(cs).szs
			assert(r1 === r2)
		}
		return r1
	}

	function thm(a) {
		var cs = []
		cnf.convert([a], cs)
		assert(sat(cs) === 'Satisfiable')

		var cs = []
		cnf.convert([etc.mk('!', a)], cs)
		assert(sat(cs) === 'Unsatisfiable')
	}

	var a = { o: 'fn', type: 'boolean' }
	var b = { o: 'fn', type: 'boolean' }

	thm(true)
	thm(etc.mk('&&', true, true, true))
	thm(etc.mk('||', false, false, true))
	thm(etc.mk('<=>', a, a))

	var p1 = { o: 'fn', name: 'p1', type: 'boolean' }
	var p2 = { o: 'fn', name: 'p2', type: 'boolean' }
	var p3 = { o: 'fn', name: 'p3', type: 'boolean' }

	thm(etc.mk('<=>', p1, etc.mk('<=>', p2, etc.mk('<=>', p1, p2))))

	function eqv(a, b) {
		thm(etc.mk('<=>', a, b))
	}

	eqv(etc.mk('!', etc.mk('!', a)), a)
	eqv(etc.mk('&&', a, b), etc.mk('&&', b, a))
	eqv(etc.mk('||', a, b), etc.mk('||', b, a))
	eqv(etc.mk('<=>', a, b), etc.mk('<=>', b, a))
	eqv(etc.mk('!', etc.mk('<=>', a, b)), etc.mk('!', etc.mk('<=>', b, a)))

	var a = { o: 'fn', type: 'individual' }
	var b = { o: 'fn', type: 'individual' }
	var f1 = { o: 'fn', type: ['individual', 'individual'] }
	var f2 = { o: 'fn', type: ['individual', 'individual', 'individual'] }
	var g1 = { o: 'fn', type: ['individual', 'individual'] }
	var g2 = { o: 'fn', type: ['individual', 'individual', 'individual'] }
	var x = { o: 'var', type: 'individual' }
	var y = { o: 'var', type: 'individual' }
	var z = { o: 'var', type: 'individual' }

	function imp(a, b) {
		return etc.mk('||', etc.mk('!', a), b)
	}

	function eq(a, b) {
		return etc.mk('==', a, b)
	}

	var fa = etc.mk('call', f1, a)
	var fb = etc.mk('call', f1, b)
	var fx = etc.mk('call', f1, x)
	var fy = etc.mk('call', f1, y)

	imp(eq(a, b), eq(fa, fb))
	thm(etc.mk('all', [x, y], imp(eq(x, y), eq(fx, fy))))
}

test()

module.exports = {
	solve,
}
