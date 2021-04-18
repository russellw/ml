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

	var passive = priorityq.mk(size)
	for (var c of clauses) priorityq.push(passive, c)
	var active = []

	function clause(neg, pos) {
		var c = cnf.clause(neg, pos)
		if (etc.eq(c, [[], [true]])) return
		priorityq.push(passive, c)
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
