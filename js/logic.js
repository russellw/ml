'use strict'
const assert = require('assert')
const etc = require('./etc')

function occurs(a, b, m) {
	if (a === b) return true
	if (m.has(b)) return occurs(a, m.get(b), m)
	if (!Array.isArray(b)) return null
	for (var x of b) if (occurs(a, x, m)) return true
}

function unify(a, b, m = new Map()) {
	if (a === b) return m
	if (a.o === 'var') return unifyvar(a, b, m)
	if (b.o === 'var') return unifyvar(b, a, m)
	if (!Array.isArray(a)) return null
	if (a.o !== b.o) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length && m; i++) m = unify(a[i], b[i], m)
	return m
}

function freshvars(a, m = new Map()) {
	if (a.o === 'var') {
		if (m.has(a)) return m.get(a)
		var x = { o: 'var', type: a.type }
		m.set(a, x)
		return x
	}
	if (!Array.isArray(a)) return a
	return etc.map(a, (b) => freshvars(b, m))
}

function match(a, b, m = new Map()) {
	if (a === b) return m
	if (a.o === 'var') {
		if (m.has(a)) return match(m.get(a), b, m)
		m.set(a, b)
		return m
	}
	if (!Array.isArray(a)) return null
	if (a.o !== b.o) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length && m; i++) m = match(a[i], b[i], m)
	return m
}

function unifyvar(a, b, m) {
	if (m.has(a)) return unify(m.get(a), b, m)
	if (m.has(b)) return unify(a, m.get(b), m)
	if (occurs(a, b, m)) return
	m.set(a, b)
	return m
}

function freevars(a) {
	var free = new Set()

	function rec(bound, a) {
		switch (a.o) {
			case 'var':
				if (!bound.has(a)) free.add(a)
				return
			case 'all':
			case 'exists':
				bound = new Set(bound)
				for (var x of a[0]) bound.add(x)
				rec(bound, a[1])
				return
		}
		if (!Array.isArray(a)) return
		for (var b of a) rec(bound, b)
	}

	rec(new Set(), a)
	return free
}

function test() {
	// fn
	var a = {}
	var b = {}
	assert(etc.eq(a, a))
	assert(!etc.eq(a, b))

	// variable
	var x = { o: 'var' }
	var y = { o: 'var' }
	var z = { o: 'var' }
	assert(!Array.isArray(x))
	assert(etc.eq(x, x))
	assert(etc.eq(y, y))
	assert(!etc.eq(x, y))
	assert(x === x)
	assert(x !== y)
	var xs = new Set()
	xs.add(x)
	assert(xs.has(x))
	assert(!xs.has(y))

	// term
	assert(etc.eq(etc.mk('&&', true, true), etc.mk('&&', true, true)))
	assert(etc.eq(etc.mk('&&', true, true), etc.mk('&&', ...[true, true])))
	assert(!etc.eq(etc.mk('&&', true, true), etc.mk('||', true, true)))
	assert(!etc.eq(etc.mk('&&', true, true), etc.mk('&&', true, false)))
	assert(!etc.eq(etc.mk('&&', true, true), x))

	// arrays
	assert(!etc.eq([true, true], x))
	assert(!etc.eq([true, true], true))

	// call
	var f = {}
	var g = {}
	assert(etc.eq(etc.mk('call', f, 1n, 2n), etc.mk('call', f, 1n, 2n)))
	assert(!etc.eq(etc.mk('call', f, 1n, 2n), etc.mk('call', g, 1n, 2n)))
	assert(!etc.eq(etc.mk('call', f, 1n, 2n), etc.mk('call', f, 1n, 3n)))

	// https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
	var m

	// Succeeds. (tautology)
	m = new Map()
	assert(unify(a, a, m))
	assert(m.size === 0)

	// a and b do not match
	m = new Map()
	assert(!unify(a, b, m))

	// Succeeds. (tautology)
	m = new Map()
	assert(unify(x, x, m))
	assert(m.size === 0)

	// x is unified with the constant a
	m = new Map()
	assert(unify(a, x, m))
	assert(m.size === 1)
	assert(etc.eq(etc.replace(x, m), a))

	// x and y are aliased
	m = new Map()
	assert(unify(x, y, m))
	assert(m.size === 1)
	assert(etc.eq(etc.replace(x, m), etc.replace(y, m)))

	// function and constant symbols match, x is unified with the constant b
	m = new Map()
	assert(unify(etc.mk('call', f, a, x), etc.mk('call', f, a, b), m))
	assert(m.size === 1)
	assert(etc.eq(etc.replace(x, m), b))

	// f and g do not match
	m = new Map()
	assert(!unify(etc.mk('call', f, a), etc.mk('call', g, a), m))

	// x and y are aliased
	m = new Map()
	assert(unify(etc.mk('call', f, x), etc.mk('call', f, y), m))
	assert(m.size === 1)
	assert(etc.eq(etc.replace(x, m), etc.replace(y, m)))

	// f and g do not match
	m = new Map()
	assert(!unify(etc.mk('call', f, x), etc.mk('call', g, y), m))

	// Fails. The f function symbols have different arity
	m = new Map()
	assert(!unify(etc.mk('call', f, x), etc.mk('call', f, y, z), m))

	// Unifies y with the term g(x)
	m = new Map()
	assert(unify(etc.mk('call', f, etc.mk('call', g, x)), etc.mk('call', f, y), m))
	assert(m.size === 1)
	assert(etc.eq(etc.replace(y, m), etc.mk('call', g, x)))

	// Unifies x with constant a, and y with the term g(a)
	m = new Map()
	assert(unify(etc.mk('call', f, etc.mk('call', g, x), x), etc.mk('call', f, y, a), m))
	assert(m.size === 2)
	assert(etc.eq(etc.replace(x, m), a))
	assert(etc.eq(etc.replace(y, m), etc.mk('call', g, a)))

	// Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
	m = new Map()
	assert(!unify(x, etc.mk('call', f, x), m))

	// Both x and y are unified with the constant a
	m = new Map()
	assert(unify(x, y, m))
	assert(unify(y, a, m))
	assert(m.size === 2)
	assert(etc.eq(etc.replace(x, m), a))
	assert(etc.eq(etc.replace(y, m), a))

	// As above (order of equations in set doesn't matter)
	m = new Map()
	assert(unify(a, y, m))
	assert(unify(x, y, m))
	assert(m.size === 2)
	assert(etc.eq(etc.replace(x, m), a))
	assert(etc.eq(etc.replace(y, m), a))

	// Fails. a and b do not match, so x can't be unified with both
	m = new Map()
	assert(unify(x, a, m))
	assert(!unify(b, x, m))

	// match is a subset of unify where only the first parameter is checked for variables
	// gives different results in several cases
	// in particular, has no notion of an occurs check
	// assumes the inputs have disjoint variables

	// Succeeds. (tautology)
	m = new Map()
	assert(match(a, a, m))
	assert(m.size === 0)

	// a and b do not match
	m = new Map()
	assert(!match(a, b, m))

	// Succeeds. (tautology)
	m = new Map()
	assert(match(x, x, m))
	assert(m.size === 0)

	// x is unified with the constant a
	// different result for match!
	m = new Map()
	assert(!match(a, x, m))

	// x and y are aliased
	m = new Map()
	assert(match(x, y, m))
	assert(m.size === 1)
	assert(etc.eq(etc.replace(x, m), etc.replace(y, m)))

	// function and constant symbols match, x is unified with the constant b
	m = new Map()
	assert(match(etc.mk('call', f, a, x), etc.mk('call', f, a, b), m))
	assert(m.size === 1)
	assert(etc.eq(etc.replace(x, m), b))

	// f and g do not match
	m = new Map()
	assert(!match(etc.mk('call', f, a), etc.mk('call', g, a), m))

	// x and y are aliased
	m = new Map()
	assert(match(etc.mk('call', f, x), etc.mk('call', f, y), m))
	assert(m.size === 1)
	assert(etc.eq(etc.replace(x, m), etc.replace(y, m)))

	// f and g do not match
	m = new Map()
	assert(!match(etc.mk('call', f, x), etc.mk('call', g, y), m))

	// Fails. The f function symbols have different arity
	m = new Map()
	assert(!match(etc.mk('call', f, x), etc.mk('call', f, y, z), m))

	// Unifies y with the term g(x)
	// different result for match!
	m = new Map()
	assert(!match(etc.mk('call', f, etc.mk('call', g, x)), etc.mk('call', f, y), m))

	// Unifies x with constant a, and y with the term g(a)
	// different result for match!
	m = new Map()
	assert(!match(etc.mk('call', f, etc.mk('call', g, x), x), etc.mk('call', f, y, a), m))

	// Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
	// not valid for match!

	// Both x and y are unified with the constant a
	m = new Map()
	assert(match(x, y, m))
	assert(match(y, a, m))
	assert(m.size === 2)
	assert(etc.eq(etc.replace(x, m), a))
	assert(etc.eq(etc.replace(y, m), a))

	// As above (order of equations in set doesn't matter)
	// different result for match!
	m = new Map()
	assert(!match(a, y, m))

	// Fails. a and b do not match, so x can't be unified with both
	m = new Map()
	assert(match(x, a, m))
	assert(!match(b, x, m))

	// freevars
	var s = freevars(etc.mk('call', f, x, y))
	assert(s.size === 2)
	assert(s.has(x))
	assert(s.has(y))

	var s = freevars(etc.mk('all', [x], etc.mk('call', f, x, y)))
	assert(s.size === 1)
	assert(s.has(y))

	// freshvars
	var y = freshvars(x)
	assert(y.o === 'var')
	assert(y !== x)

	var b = freshvars(5)
	assert(b === 5)

	var b = freshvars(etc.mk('call', f, x, y))
	assert(b.o === 'call')
	assert(b[0] === f)
	var x1 = b[1]
	assert(x1.o === 'var')
	assert(x1 !== x && x1 !== y)
	var y1 = b[2]
	assert(y1.o === 'var')
	assert(y1 !== x && y1 !== y)
	assert(x1 !== y1)
}

test()

exports.unify = unify
exports.match = match
exports.freevars = freevars
