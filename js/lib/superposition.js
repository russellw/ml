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

	var passive = new priorityq(size)
	for (var c of clauses) {
		var d = cnf.simplify(c)
		if (etc.eq(d, c)) d = c
		else {
			d.how = 'simplify'
			d.from = [c]
		}
		passive.push(d)
	}

	function push(c, m, how, ...from) {
		c = cnf.simplify(c, m)
		if (etc.eq(c, [[], [true]])) return
		c.how = how
		c.from = from
		passive.push(c)
	}

	var active = []

	// equality resolution
	// c | c0 != c1
	// ->
	// c/s
	// where
	// s = unify(c0, c1)

	// push new clause
	function resolvep(c, ci, m) {
		var neg = [...c[0]]
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

		var neg = [...c[0]]
		neg.push(equate(c1, d1))

		var pos = [...c[1]]
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

		var neg = [...d[0]]
		neg.splice(di, 1)
		neg = c[0].concat(neg)
		neg.push(equate(splice(d0, path, c1), d1))

		var pos = [...c[1]]
		pos.splice(ci, 1)
		pos = pos.concat(d[1])

		push([neg, pos], m, 'ns', c, d)
	}

	// descend into subterms
	function nsuperpositiond(c, d, ci, c0, c1, di, d0, d1, path, a) {
		if (a.o === 'var') return
		nsuperpositionp(c, d, ci, c0, c1, di, d0, d1, path, a)
		if (!Array.isArray(a)) return
		for (var i = 0; i < a.length; i++) nsuperpositiond(c, d, ci, c0, c1, di, d0, d1, path.concat(i), a[i])
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

		var cpos = [...c[1]]
		cpos.splice(ci, 1)
		var dpos = [...d[1]]
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
		for (var i = 0; i < a.length; i++) psuperpositiond(c, d, ci, c0, c1, di, d0, d1, path.concat(i), a[i])
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
		var g = passive.pop()

		// no more clauses => we are done, proof not found
		if (!g) {
			if (complete) return { szs: 'Satisfiable', active }
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
	var x = { o: 'var', type: 'individual', name: 'x' }
	var y = { o: 'var', type: 'individual', name: 'y' }
	var z = { o: 'var', type: 'individual', name: 'z' }

	var c = [[p], []]
	var r = solve([c])
	assert(r.szs === 'Satisfiable')

	var c = [[], [p]]
	var r = solve([c])
	assert(r.szs === 'Satisfiable')

	// from SYN014-2
	// used clauses:
	// symmetryish
	// c_20
	// transitivityish
	// c_19
	// c_23
	var n = { o: 'fn', type: 'individual' }
	var k = { o: 'fn', type: 'individual' }
	var M = { o: 'fn', type: 'individual' }
	var equalish = { o: 'fn', type: ['boolean', 'individual', 'individual'] }

	var symmetryish = [[etc.mk('call', equalish, x, y)], [etc.mk('call', equalish, y, x)]]
	var c_20 = [[], [etc.mk('call', equalish, n, k)]]
	var r = solve([symmetryish, c_20])
	assert(r.szs === 'Satisfiable')
	var c1 = [[], [etc.mk('call', equalish, k, n)]]
	var found = false
	for (var c of r.active) if (etc.eq(c, c1)) found = true
	assert(found)

	var x = { o: 'var', type: 'individual', name: 'x' }
	var y = { o: 'var', type: 'individual', name: 'y' }
	var z = { o: 'var', type: 'individual', name: 'z' }
	var transitivityish = [[etc.mk('call', equalish, x, y), etc.mk('call', equalish, y, z)], [etc.mk('call', equalish, x, z)]]
	var c2 = [[etc.mk('call', equalish, x, k)], [etc.mk('call', equalish, x, n)]]
	/*
	var r = solve([transitivityish, c1])
	assert(r.szs === 'Satisfiable')
	var found = false
	for (var c of r.active) if (etc.isomorphic(c, c2)) found = true
	assert(found)
	*/

	var c_19 = [[etc.mk('call', equalish, M, n)], []]
	var c_23 = [[], [etc.mk('call', equalish, M, k)]]
	var r = solve([c2, c_19, c_23])
	assert(r.szs === 'Unsatisfiable')

	// now try with reflexivityish
	var x = { o: 'var', type: 'individual', name: 'xr' }
	var reflexivityish = [[], [etc.mk('call', equalish, x, x)]]

	var m = subsumption.subsumes(reflexivityish, symmetryish)
	if (m) etc.show(m)
	assert(!m)

	var r = solve([reflexivityish, symmetryish, c_20])
	assert(r.szs === 'Satisfiable')
	var c1 = [[], [etc.mk('call', equalish, k, n)]]
	var found = false
	for (var c of r.active) if (etc.isomorphic(c, c1)) found = true
	assert(found)

	/*
	var r = solve([reflexivityish, transitivityish, c1])
	assert(r.szs === 'Satisfiable')
	var x = { o: 'var', type: 'individual' }
	var c2 = [[etc.mk('call', equalish, x, k)], [etc.mk('call', equalish, x, n)]]
	var found = false
	for (var c of r.active) if (etc.isomorphic(c, c2)) found = true
	assert(found)
	*/

	// c2  : ~ equalish(X,k) | equalish(X,n)
	// c_19: ~ equalish(m,n)
	// c_23: equalish(m,k)
	var c_19 = [[etc.mk('call', equalish, M, n)], []]
	var c_23 = [[], [etc.mk('call', equalish, M, k)]]

	// assert(!subsumption.subsumes(reflexivityish,c2))
	assert(!subsumption.subsumes(reflexivityish, c_19))
	assert(!subsumption.subsumes(reflexivityish, c_23))

	var r = solve([reflexivityish, c2, c_19, c_23])
	assert(r.szs === 'Unsatisfiable')
}

test()

module.exports = {
	solve,
}
