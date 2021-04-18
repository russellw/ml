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
	for (;;) {
		// given clause
		var g = priorityq.pop(passive)

		// no more clauses => we are done, proof not found
		if (!g) {
			if (complete) return { sat: true }
			return { szs: 'GaveUp' }
		}

		// empty (false) clause => proof found
		if (etc.eq(g, [[], []]))
			return {
				sat: false,
				proof: g,
			}
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
