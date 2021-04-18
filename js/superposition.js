'use strict'
const cnf = require('./cnf')
const etc = require('./etc')
const subsumption = require('./subsumption')
const priorityq = require('./priorityq')
const assert = require('assert')

function size(a) {
	if (!Array.isArray(a)) return 1
	var n = 0
	for (var b of a) n += size(b)
	return n
}

function solve(clauses) {
	var complete = true

	function push(c, m) {
		c = cnf.simplify(c, m)
		if (etc.eq(c, [[], [true]])) return
		priorityq.push(passive, c)
	}

	var passive = priorityq.mk(size)
	for (var c of clauses) push(c, new Map())
	var active = []

	/*
	equality resolution
		c | c0 !== c1
	->
		c/s
	where
		s = unify(c0, c1)
	*/

	// substitute and make new clause
	function resolveq(c, ci, m) {
		var neg = []
		for (var i = 0; i < c[0].length; i++) if (i !== ci) neg.push(c[0][i])
	}

	// for each negative equation
	function resolve(c) {
		for (var i = 0; i < c[0].length; i++) {
			var ce = etc.eqn(c[0][i])
			var m = logic.unify(ce[0], e[1])
			if (m) resolveq(c, ci, m)
		}
	}

	// saturation proof procedure tries to perform all possible derivations until it derives false
	loop: for (;;) {
		// given clause
		var g = priorityq.pop(passive)

		// no more clauses => we are done, proof not found
		if (!g) {
			if (complete) return { sat: true }
			return { szs: 'GaveUp' }
		}

		// empty (false) clause => proof found
		if (etc.eq(g, [[], []])) return { sat: false, proof: g }

		// algorithms being used here, assume clauses have distinct variable names
		var h = logic.freshvars(g)

		// this is the Discount loop
		// in which only active clauses participate in subsumption checks
		// in tests, it performed slightly better than the Otter loop
		// in which passive clauses also participate

		// forward subsumption
		for (var c of active) {
			if (c.subsumed) continue
			if (subsumption.subsumes(c, h)) continue loop
		}

		// backward subsumption
		for (var c of active) {
			if (c.subsumed) continue
			if (subsumption.subsumes(h, c)) c.subsumed = true
		}

		// add g to active clauses before inference
		// because we will sometimes need to combine g
		// with (the fresh-variable version of) itself
		active.push(g)

		// infer
	}
}

function test() {
	assert(size(5) === 1)
	assert(size(etc.mk('==', etc.mk('unary-', 10), etc.mk('+', 11, 12))), 3)

	assert(solve([]).sat === true)
	assert(solve([[[], []]]).sat === false)
}

test()

exports.solve = solve
