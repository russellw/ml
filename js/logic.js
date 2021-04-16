'use strict'
const assert = require('assert')

function occurs(a, b, m) {
	if (a === b) return true
	if (m.has(b)) return occurs(a, m.get(b), m)
	if (!Array.isArray(b)) return null
	for (var x of b) if (occurs(a, x, m)) return true
}

function unify(a, b, m = new Map()) {
	if (a === b) return m
	if (a.op === 'var') return unifyvar(a, b, m)
	if (b.op === 'var') return unifyvar(b, a, m)
	if (!Array.isArray(a)) return null
	if (a.op !== b.op) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length && m; i++) m = unify(a[i], b[i], m)
	return m
}

function simplify(a, m = new Map()) {
	if (m.has(a)) return simplify(m.get(a), m)
	return map(a, (b) => simplify(b, m))
}

function match(a, b, m = new Map()) {
	if (a === b) return m
	if (a.op === 'var') {
		if (m.has(a)) return match(m.get(a), b, m)
		m.set(a, b)
		return m
	}
	if (!Array.isArray(a)) return null
	if (a.op !== b.op) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length && m; i++) m = unify(a[i], b[i], m)
	return m
}

function unifyvar(a, b, m) {
	if (m.has(a)) return unify(m.get(a), b, m)
	if (m.has(b)) return unify(a, m.get(b), m)
	if (occurs(a, b, m)) return
	m.set(a, b)
	return m
}

function term(op, ...args) {
	var a = Array.from(args)
	a.op = op
	return a
}

function eq(a, b) {
	if (a === b) return true
	if (!Array.isArray(a)) return
	if (a.op !== b.op) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length; i++) if (!eq(a[i], b[i])) return
	return true
}

function replace(a, m) {
	if (m.has(a)) return replace(m.get(a), m)
	return map(a, (b) => replace(b, m))
}

function map(a, f) {
	if (!Array.isArray(a)) return a
	var r = []
	Object.assign(r, a)
	for (var i = 0; i < r.length; i++) r[i] = f(r[i])
	return r
}

// map
assert(
	eq(
		map(term('+', 1, 2), (a) => a + 10),
		term('+', 11, 12)
	)
)

// bool
assert(eq(false, false))
assert(eq(true, true))
assert(!eq(false, true))

// integer
assert(eq(0n, 0n))
assert(!eq(1_000_000_000_000_000_000_000_000n, 1_000_000_000_000_000_000_000_001n))

// fn
var a = {}
var b = {}
assert(eq(a, a))
assert(!eq(a, b))

// variable
var x = { op: 'var' }
var y = { op: 'var' }
var z = { op: 'var' }
assert(eq(x, x))
assert(eq(y, y))
assert(!eq(x, y))

// term
assert(eq(term('&&', true, true), term('&&', true, true)))
assert(eq(term('&&', true, true), term('&&', ...[true, true])))
assert(!eq(term('&&', true, true), term('||', true, true)))
assert(!eq(term('&&', true, true), term('&&', true, false)))
assert(!eq(term('&&', true, true), x))

// arrays
assert(!eq([true, true], x))
assert(!eq([true, true], true))

// call
var f = {}
var g = {}
assert(eq(term('call', f, 1n, 2n), term('call', f, 1n, 2n)))
assert(!eq(term('call', f, 1n, 2n), term('call', g, 1n, 2n)))
assert(!eq(term('call', f, 1n, 2n), term('call', f, 1n, 3n)))

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
assert(eq(replace(x, m), a))

// x and y are aliased
m = new Map()
assert(unify(x, y, m))
assert(m.size === 1)
assert(eq(replace(x, m), replace(y, m)))

// function and constant symbols match, x is unified with the constant b
m = new Map()
assert(unify(term('call', f, a, x), term('call', f, a, b), m))
assert(m.size === 1)
assert(eq(replace(x, m), b))

// f and g do not match
m = new Map()
assert(!unify(term('call', f, a), term('call', g, a), m))

// x and y are aliased
m = new Map()
assert(unify(term('call', f, x), term('call', f, y), m))
assert(m.size === 1)
assert(eq(replace(x, m), replace(y, m)))

// f and g do not match
m = new Map()
assert(!unify(term('call', f, x), term('call', g, y), m))

// Fails. The f function symbols have different arity
m = new Map()
assert(!unify(term('call', f, x), term('call', f, y, z), m))

// Unifies y with the term g(x)
m = new Map()
assert(unify(term('call', f, term('call', g, x)), term('call', f, y), m))
assert(m.size === 1)
assert(eq(replace(y, m), term('call', g, x)))

// Unifies x with constant a, and y with the term g(a)
m = new Map()
assert(unify(term('call', f, term('call', g, x), x), term('call', f, y, a), m))
assert(m.size === 2)
assert(eq(replace(x, m), a))
assert(eq(replace(y, m), term('call', g, a)))

// Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
m = new Map()
assert(!unify(x, term('call', f, x), m))

// Both x and y are unified with the constant a
m = new Map()
assert(unify(x, y, m))
assert(unify(y, a, m))
assert(m.size === 2)
assert(eq(replace(x, m), a))
assert(eq(replace(y, m), a))

// As above (order of equations in set doesn't matter)
m = new Map()
assert(unify(a, y, m))
assert(unify(x, y, m))
assert(m.size === 2)
assert(eq(replace(x, m), a))
assert(eq(replace(y, m), a))

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
assert(eq(replace(x, m), replace(y, m)))

// function and constant symbols match, x is unified with the constant b
m = new Map()
assert(match(term('call', f, a, x), term('call', f, a, b), m))
assert(m.size === 1)
assert(eq(replace(x, m), b))

// f and g do not match
m = new Map()
assert(!match(term('call', f, a), term('call', g, a), m))

// x and y are aliased
m = new Map()
assert(match(term('call', f, x), term('call', f, y), m))
assert(m.size === 1)
assert(eq(replace(x, m), replace(y, m)))

// f and g do not match
m = new Map()
assert(!match(term('call', f, x), term('call', g, y), m))

// Fails. The f function symbols have different arity
m = new Map()
assert(!match(term('call', f, x), term('call', f, y, z), m))

// Unifies y with the term g(x)
m = new Map()
assert(match(term('call', f, term('call', g, x)), term('call', f, y), m))
assert(m.size === 1)
assert(eq(replace(y, m), term('call', g, x)))

// Unifies x with constant a, and y with the term g(a)
m = new Map()
assert(match(term('call', f, term('call', g, x), x), term('call', f, y, a), m))
assert(m.size === 2)
assert(eq(replace(x, m), a))
assert(eq(replace(y, m), term('call', g, a)))

// Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
// not valid for match!

// Both x and y are unified with the constant a
m = new Map()
assert(match(x, y, m))
assert(match(y, a, m))
assert(m.size === 2)
assert(eq(replace(x, m), a))
assert(eq(replace(y, m), a))

// As above (order of equations in set doesn't matter)
// different result for match!
m = new Map()
assert(!match(a, y, m))

// Fails. a and b do not match, so x can't be unified with both
m = new Map()
assert(match(x, a, m))
assert(!match(b, x, m))

// simplify
assert(eq(simplify(x), x))
m = new Map()
m.set(x, y)
assert(eq(simplify(x, m), y))
assert(eq(simplify(term('call', f, x, y), m), term('call', f, y, y)))

// exports
exports.occurs = occurs
exports.unify = unify
exports.eq = eq
exports.term = term
exports.simplify = simplify
