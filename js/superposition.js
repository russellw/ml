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
	if (etc.type(a) === 'boolean') return a === true || b === true
	return true
}

function equate(a, b) {
	assert(equatable(a, b))
	if (a === true) return b
	if (b === true) return a
	return etc.mk('==', a, b)
}

function splice(a, path, b, i = 0) {
	if (i === path.length) return b
	assert(Array.isArray(a))
	var r = []
	Object.assign(r, a)
	r[path[i]] = splice(r[path[i]], path, b, i + 1)
	return r
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

	// equality resolution
	// c | c0 != c1
	// ->
	// c/s
	// where
	// s = unify(c0, c1)

	// push new clause
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

	// equality factoring
	// c | c0 = c1 | d0 = d1
	// ->
	// (c | c0 = c1 | c1 != d1)/s
	// where
	// s = unify(c0, d0)

	// check and push new clause
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

	// negative superposition
	// c | c0 = c1, d | d0(a) != d1
	// ->
	// (c | d | d0(c1) != d1)/m
	// where
	// m = unify(c0, a)
	// a is not a variable

	// check and push new clause
	function nsuperpositionp(c, d, ci, c0, c1, di, d0, d1, path, a) {
		var m = etc.unify(c0, a)
		if (!m) return

		var neg = d[0].slice()
		neg.splice(di, 1)
		neg = c[0].concat(neg)
		neg.push(equate(splice(d0, path, c1), d1))

		var pos = c[1].slice()
		pos.splice(ci, 1)
		pos = pos.concat(d[1])
	}

	// descend into subterms
	function nsuperpositiond(c, d, ci, c0, c1, di, d0, d1, path, a) {
		if (a.o === 'var') return
		nsuperpositionp(c, d, ci, c0, c1, di, d0, d1, path, a)
		if (!Array.isArray(a)) return
		for (var i = 0; i < a.length; i++) {
			path.push(i)
			nsuperpositiond(c, d, ci, c0, c1, di, d0, d1, path, a[i])
			path.pop()
		}
	}

	// for each negative equation in d (both directions)
	function nsuperposition1(c, d, ci, c0, c1) {
		if (c0 === true) return
		for (var i = 0; i < d[0].length; i++) {
			var e = etc.eqn(d[0][i])
			nsuperpositiond(c, d, ci, c0, c1, i, e[0], e[1])
			nsuperpositiond(c, d, ci, c0, c1, i, e[1], e[0])
		}
	}

	// for each positive equation in c (both directions)
	function nsuperposition(c, d) {
		for (var i = 0; i < c[1].length; i++) {
			var e = etc.eqn(c[1][i])
			nsuperposition1(c, d, i, e[0], e[1])
			nsuperposition1(c, d, i, e[1], e[0])
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
		for (var c of active) {
			nsuperposition(c, h)
			nsuperposition(h, c)
		}
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

	var a = 1
	var path = []
	var b = 2
	var r = 2
	assert(etc.eq(splice(a, path, b), r))

	var a = etc.mk('+', 1, 2)
	var path = [0]
	var b = 3
	var r = etc.mk('+', 3, 2)
	assert(etc.eq(splice(a, path, b), r))
}

test()

exports.solve = solve
