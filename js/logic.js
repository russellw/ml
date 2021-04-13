'use strict'
var assert = require('assert')

function occurs(a, b, m) {
	if (a === b) return true
	if (m.has(b)) return occurs(a, m.get(b), m)
	if (!b.length) return
	for (var x of b) if (occurs(a, x, m)) return true
}

function unify(a, b, m = new Map()) {
	if (a === b) return m
	if (a.op === 'variable') return unifyVariable(a, b, m)
	if (b.op === 'variable') return unifyVariable(b, a, m)
	if (a.op !== b.op) return
	if (!a.length) return eq(a, b) ? m : null
	if (a.f !== b.f) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length && m; i++) m = unify(a[i], b[i], m)
	return m
}

function simplify(a, m = new Map()) {
	if (m.has(a)) return simplify(m.get(a), m)
	if (!a.length) return a
	var r = []
	Object.assign(r, a)
	for (var i = 0; i < r.length; i++) r[i] = simplify(r[i], m)
	return r
}

function match(a, b, m = new Map()) {
	if (a === b) return m
	if (a.op === 'variable') {
		if (m.has(a)) return match(m.get(a), b, m)
		m.set(a, b)
		return m
	}
	if (a.op !== b.op) return
	if (!a.length) return eq(a, b) ? m : null
	if (a.f !== b.f) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length && m; i++) m = unify(a[i], b[i], m)
	return m
}

function unifyVariable(a, b, m) {
	if (m.has(a)) return unify(m.get(a), b, m)
	if (m.has(b)) return unify(a, m.get(b), m)
	if (occurs(a, b, m)) return
	m.set(a, b)
	return m
}

function bool(val) {
	assert(typeof val === 'boolean')
	var a = []
	a.op = 'bool'
	a.val = val
	return a
}

function call(f, ...args) {
	var a = Array.from(args)
	a.op = 'call'
	a.f = f
	return a
}

function distinctObj(name) {
	assert(typeof name === 'string')
	var a = []
	a.name = name
	a.op = 'distinctObj'
	return a
}

function fn(name) {
	var a = []
	a.name = name
	a.op = 'fn'
	return a
}

function term(op, ...args) {
	switch (op) {
		case '<':
		case '<=':
		case '<=>':
		case '==':
		case '=>':
		case '>':
		case '>=':
			assert(args.length === 2)
			break
		case '!':
			assert(args.length === 1)
			break
	}
	var a = Array.from(args)
	a.op = op
	return a
}

function variable(name) {
	var a = []
	a.name = name
	a.op = 'variable'
	return a
}

function integer(val) {
	switch (typeof val) {
		case 'number':
		case 'string':
			val = BigInt(val)
			break
	}
	var a = []
	a.op = 'integer'
	a.val = val
	return a
}

function eq(a, b) {
	if (a === b) return true
	if (a.op !== b.op) return
	switch (a.op) {
		case 'integer':
		case 'bool':
			return a.val === b.val
		case 'call':
			if (a.f !== b.f) return
			break
		case 'distinctObj':
			return a.name === b.name
		case 'fn':
		case 'variable':
			return
	}
	if (a.length !== b.length) return
	for (var i = 0; i < a.length; i++) if (!eq(a[i], b[i])) return
	return true
}

function replace(a, m) {
	if (m.has(a)) return replace(m.get(a), m)
	if (!a.length) return a
	var r = []
	Object.assign(r, a)
	for (var i = 0; i < r.length; i++) r[i] = replace(r[i], m)
	return r
}

//bool
assert(eq(bool(false), bool(false)))
assert(eq(bool(true), bool(true)))
assert(!eq(bool(false), bool(true)))

//integer
assert(eq(integer(0), integer(0)))
assert(!eq(integer(0), integer(1)))
assert(eq(integer(1_000_000_000_000_000_000_000_000n), integer(1_000_000_000_000n * 1_000_000_000_000n)))
assert(eq(integer(1_000_000_000_000_000_000_000_000n), integer('1000000000000000000000000')))
assert(eq(integer(1_000_000_000_000_000_000_000_000n), integer('+1000000000000000000000000')))
assert(eq(integer(-1_000_000_000_000_000_000_000_000n), integer('-1000000000000000000000000')))

//distinct object
assert(eq(distinctObj('a'), distinctObj('a')))
assert(!eq(distinctObj('a'), distinctObj('b')))

//fn
assert(!eq(fn('a'), fn('a')))
var a = fn('a')
var b = fn()
assert(eq(a, a))
assert(eq(b, b))
assert(!eq(a, b))

//variable
assert(!eq(variable('x'), variable('x')))
var x = variable('x')
var y = variable()
var z = variable()
assert(eq(x, x))
assert(eq(y, y))
assert(!eq(x, y))

//term
assert(eq(term('&&', bool(true), bool(true)), term('&&', bool(true), bool(true))))
assert(eq(term('&&', bool(true), bool(true)), term('&&', ...[bool(true), bool(true)])))
assert(!eq(term('&&', bool(true), bool(true)), term('||', bool(true), bool(true))))
assert(!eq(term('&&', bool(true), bool(true)), term('&&', bool(true), bool(false))))

//call
var f = fn('f')
var g = fn('g')
assert(eq(call(f, integer(1), integer(2)), call(f, ...[integer(1), integer(2)])))
assert(!eq(call(f, integer(1), integer(2)), call(g, integer(1), integer(2))))
assert(!eq(call(f, integer(1), integer(2)), call(f, integer(1), integer(3))))

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
assert(unify(call(f, a, x), call(f, a, b), m))
assert(m.size === 1)
assert(eq(replace(x, m), b))

// f and g do not match
m = new Map()
assert(!unify(call(f, a), call(g, a), m))

// x and y are aliased
m = new Map()
assert(unify(call(f, x), call(f, y), m))
assert(m.size === 1)
assert(eq(replace(x, m), replace(y, m)))

// f and g do not match
m = new Map()
assert(!unify(call(f, x), call(g, y), m))

// Fails. The f function symbols have different arity
m = new Map()
assert(!unify(call(f, x), call(f, y, z), m))

// Unifies y with the term g(x)
m = new Map()
assert(unify(call(f, call(g, x)), call(f, y), m))
assert(m.size === 1)
assert(eq(replace(y, m), call(g, x)))

// Unifies x with constant a, and y with the term g(a)
m = new Map()
assert(unify(call(f, call(g, x), x), call(f, y, a), m))
assert(m.size === 2)
assert(eq(replace(x, m), a))
assert(eq(replace(y, m), call(g, a)))

// Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
m = new Map()
assert(!unify(x, call(f, x), m))

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
//different result for match!
m = new Map()
assert(!match(a, x, m))

// x and y are aliased
m = new Map()
assert(match(x, y, m))
assert(m.size === 1)
assert(eq(replace(x, m), replace(y, m)))

// function and constant symbols match, x is unified with the constant b
m = new Map()
assert(match(call(f, a, x), call(f, a, b), m))
assert(m.size === 1)
assert(eq(replace(x, m), b))

// f and g do not match
m = new Map()
assert(!match(call(f, a), call(g, a), m))

// x and y are aliased
m = new Map()
assert(match(call(f, x), call(f, y), m))
assert(m.size === 1)
assert(eq(replace(x, m), replace(y, m)))

// f and g do not match
m = new Map()
assert(!match(call(f, x), call(g, y), m))

// Fails. The f function symbols have different arity
m = new Map()
assert(!match(call(f, x), call(f, y, z), m))

// Unifies y with the term g(x)
m = new Map()
assert(match(call(f, call(g, x)), call(f, y), m))
assert(m.size === 1)
assert(eq(replace(y, m), call(g, x)))

// Unifies x with constant a, and y with the term g(a)
m = new Map()
assert(match(call(f, call(g, x), x), call(f, y, a), m))
assert(m.size === 2)
assert(eq(replace(x, m), a))
assert(eq(replace(y, m), call(g, a)))

// Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
//not valid for match!

// Both x and y are unified with the constant a
m = new Map()
assert(match(x, y, m))
assert(match(y, a, m))
assert(m.size === 2)
assert(eq(replace(x, m), a))
assert(eq(replace(y, m), a))

// As above (order of equations in set doesn't matter)
//different result for match!
m = new Map()
assert(!match(a, y, m))

// Fails. a and b do not match, so x can't be unified with both
m = new Map()
assert(match(x, a, m))
assert(!match(b, x, m))

//simplify
assert(eq(simplify(x), x))
m = new Map()
m.set(x, y)
assert(eq(simplify(x, m), y))
assert(eq(simplify(call(f, x, y), m), call(f, y, y)))

//exports
exports.occurs = occurs
exports.distinctObj = distinctObj
exports.unify = unify
exports.bool = bool
exports.call = call
exports.eq = eq
exports.fn = fn
exports.term = term
exports.variable = variable
