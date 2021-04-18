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
	function resolvep(c, ci, m) {
		var neg = c[0].slice()
		neg.splice(ci, 1)
		var pos = c[1]
		push([neg, pos], m)
	}

	// for each negative equation
	function resolve(c) {
		for (var i = 0; i < c[0].length; i++) {
			var e = etc.eqn(c[0][i])
			var m = etc.unify(e[0], e[1])
			if (m) resolvep(c, i, m)
		}
	}

	/*
	equality factoring
		c | c0 = c1 | d0 = d1
	->
		(c | c0 = c1 | c1 !== d1)/s
	where
		s = unify(c0, d0)
	*/

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
		var h = etc.freshvars(g)

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
		resolve(h)
	}
}

function test() {
	assert(size(5) === 1)
	assert(size(etc.mk('==', etc.mk('unary-', 10), etc.mk('+', 11, 12))), 3)

	var r = solve([])
	assert(r.sat === true)

	var c = [[], []]
	var r = solve([c])
	assert(r.sat === false)
	assert(etc.eq(r.proof, [[], []]))

	var c = [[etc.mk('==', 1, 1)], []]
	var r = solve([c])
	assert(r.sat === false)
	assert(etc.eq(r.proof, [[], []]))
}

test()

exports.solve = solve
