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

function solve(clauses, deadline) {
	cnf.ckclauses(clauses)

	var complete = true
	for (var c of clauses)
		for (var L of c)
			for (var a of L)
				etc.walk(a, (b) => {
					if (etc.isnumtype(etc.type(b))) complete = false
				})

	var passive = priorityq.mk(size)
	for (var c of clauses) {
		var d = cnf.simplify(c)
		if (etc.eq(d, c)) d = c
		else {
			d.how = 'simplify'
			d.from = [c]
		}
		priorityq.push(passive, d)
	}
	var active = []

	function push(c, m, how, ...from) {
		c = cnf.simplify(c, m)
		if (etc.eq(c, [[], [true]])) return
		c.how = how
		c.from = from
		priorityq.push(passive, c)
	}

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

		push([neg, c[1]], m, 'resolve', c)
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

		push([neg, pos], m, 'factor', c)
	}

	// for each positive equation (both directions) again
	function factor1(c, ci, c0, c1) {
		for (var i = 0; i < c[1].length; i++) {
			if (i === ci) continue
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

		push([neg, pos], m, 'ns', c, d)
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
			nsuperpositiond(c, d, ci, c0, c1, i, e[0], e[1], [], e[0])
			nsuperpositiond(c, d, ci, c0, c1, i, e[1], e[0], [], e[1])
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

	// positive superposition
	// c | c0 = c1, d | d0(a) = d1
	// ->
	// (c | d | d0(c1) = d1)/m
	// where
	// m = unify(c0, a)
	// a is not a variable

	// check and push new clause
	function psuperpositionp(c, d, ci, c0, c1, di, d0, d1, path, a) {
		var m = etc.unify(c0, a)
		if (!m) return

		var neg = c[0].concat(d[0])

		var cpos = c[1].slice()
		cpos.splice(ci, 1)
		var dpos = d[1].slice()
		dpos.splice(di, 1)
		var pos = cpos.concat(dpos)
		pos.push(equate(splice(d0, path, c1), d1))

		push([neg, pos], m, 'ps', c, d)
	}

	// descend into subterms
	function psuperpositiond(c, d, ci, c0, c1, di, d0, d1, path, a) {
		if (a.o === 'var') return
		psuperpositionp(c, d, ci, c0, c1, di, d0, d1, path, a)
		if (!Array.isArray(a)) return
		for (var i = 0; i < a.length; i++) {
			path.push(i)
			psuperpositiond(c, d, ci, c0, c1, di, d0, d1, path, a[i])
			path.pop()
		}
	}

	// for each negative equation in d (both directions)
	function psuperposition1(c, d, ci, c0, c1) {
		if (c0 === true) return
		for (var i = 0; i < d[0].length; i++) {
			var e = etc.eqn(d[0][i])
			psuperpositiond(c, d, ci, c0, c1, i, e[0], e[1], [], e[0])
			psuperpositiond(c, d, ci, c0, c1, i, e[1], e[0], [], e[1])
		}
	}

	// for each positive equation in c (both directions)
	function psuperposition(c, d) {
		for (var i = 0; i < c[1].length; i++) {
			var e = etc.eqn(c[1][i])
			psuperposition1(c, d, i, e[0], e[1])
			psuperposition1(c, d, i, e[1], e[0])
		}
	}

	// saturation proof procedure tries to perform all possible derivations until it derives false
	loop: for (;;) {
		etc.cktime(deadline)

		// given clause
		var g = priorityq.pop(passive)

		// no more clauses => we are done, proof not found
		if (!g) {
			if (complete) return { szs: 'Satisfiable' }
			return { szs: 'GaveUp' }
		}

		// empty (false) clause => proof found
		if (etc.eq(g, [[], []])) return { szs: 'Unsatisfiable', proof: g }

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
			psuperposition(c, h)
			psuperposition(h, c)
		}
	}
}

function test() {
	assert(size(5) === 1)
	assert(size(etc.mk('==', etc.mk('unary-', 10), etc.mk('+', 11, 12))), 3)

	var r = solve([])
	assert(r.szs === 'Satisfiable')

	var c = [[], []]
	var r = solve([c])
	assert(r.szs === 'Unsatisfiable')
	assert(etc.eq(r.proof, [[], []]))

	var c = [[etc.mk('==', 1n, 1n)], []]
	var r = solve([c])
	assert(r.szs === 'Unsatisfiable')
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

	var a = { o: 'fn', type: 'individual' }
	var b = { o: 'fn', type: 'individual' }
	var f1 = { o: 'fn', type: ['individual', 'individual'] }
	var f2 = { o: 'fn', type: ['individual', 'individual', 'individual'] }
	var g1 = { o: 'fn', type: ['individual', 'individual'] }
	var g2 = { o: 'fn', type: ['individual', 'individual', 'individual'] }
	var p = { o: 'fn', type: 'boolean' }
	var p1 = { o: 'fn', type: ['boolean', 'individual'] }
	var p2 = { o: 'fn', type: ['boolean', 'individual', 'individual'] }
	var q = { o: 'fn', type: 'boolean' }
	var q1 = { o: 'fn', type: ['boolean', 'individual'] }
	var q2 = { o: 'fn', type: ['boolean', 'individual', 'individual'] }
	var x = { o: 'var', type: 'individual' }
	var y = { o: 'var', type: 'individual' }
	var z = { o: 'var', type: 'individual' }

	var c = [[p], []]
	var r = solve([c])
	assert(r.szs === 'Satisfiable')

	var c = [[], [p]]
	var r = solve([c])
	assert(r.szs === 'Satisfiable')
}

test()

exports.solve = solve
