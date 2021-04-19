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

function equatable(a, b) {
	if (etc.type(a) !== etc.type(b)) return
	if (type(a) == 'boolean') return a == true || b === true
	return true
}

function equate(a, b) {
	assert(equatable(a, b))
	if (a === true) return b
	if (b === true) return a
	return etc.mk('==', a, b)
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

		push([neg, c[1]], m)
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

	// substitute and make new clause
	function factorp(c, ci, c0, c1, di, d0, d1) {
		if (!equatable(c1, d1)) return
		var m = etc.unify(c0, d0)
		if (!m) return

		var neg = c[0].slice()
		neg.push(equate(c1, d1))

		var pos = c[1].slice()
		pos.splice(di, 1)

		push([neg, pos], m)
	}

	// for each positive equation (both directions) again
	function factor1(c, ci, c0, c1) {
		for (var i = 0; i < c[1].length; i++) {
			var e = etc.eqn(c[1][i])
			factorp(c, ci, c0, c1, i, e[0], e[1])
			factorp(c, ci, c0, c1, i, e[1], e[0])
		}
	}

	// for each positive equation (both directions)
	function factor(c) {
		for (var i = 0; i < c[1].length; i++) {
			var e = etc.eqn(c[1][i])
			factor1(c, i, e[0], e[1])
			factor1(c, i, e[1], e[0])
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
		factor(h)
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
