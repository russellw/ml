'use strict'
var assert = require('assert')

function occurs(a, b, m) {
	if (a === b) return true
	if (m.has(b)) return occurs(a, m.get(b), m)
	if (!b.args) return false
	for (var x of b.args) if (occurs(a, x, m)) return true
}

function unify(a, b, m = new Map()) {
	if (a === b) return m
	if (a.op === 'var') return unifyVar(a, b, m)
	if (b.op === 'var') return unifyVar(b, a, m)
	if (a.op !== b.op) return null
	switch (a.op) {
		case 'call':
			if (a.f !== b.f) return null
			break
		case 'const':
			return a.val === b.val
	}
	if (!a.args) return m
	if (a.args.length !== b.args.length) return null
	for (var i = 0; i < a.args.length && m; i++)
		m = unify(a.args[i], b.args[i], m)
	return m
}

function unifyVar(a, b, m) {
	if (m.has(a)) return unify(m.get(a), b, m)
	if (m.has(b)) return unify(a, m.get(b), m)
	if (occurs(a, b, m)) return null
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

function call(f, args) {
	assert(!args.op)
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
		case '!=':
		case '<':
		case '<=':
		case '<=>':
		case '<~>':
		case '=':
		case '=>':
		case '>':
		case '>=':
		case '~&':
		case '~|':
			assert(args.length === 2)
			break
		case '&':
		case '|':
			break
		case '~':
			assert(args.length === 1)
			break
		default:
			throw new Error(op)
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

function isTerm(a) {
	if (!Array.isArray(a)) return
	if (typeof a.op !== 'string') return
	return true
}

function eq(a, b) {
	assert(isTerm(a))
	assert(isTerm(b))
	if (a === b) return true
	if (a.op !== b.op) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length; i++) if (!eq(a[i], b[i])) return
	switch (a.op) {
		case 'integer':
		case 'bool':
			return a.val === b.val
		case 'call':
			return a.f === b.f
		case 'distinctObj':
			return a.name === b.name
		case 'fn':
		case 'variable':
			return
	}
	return true
}

//bool
assert(eq(bool(false), bool(false)))
assert(eq(bool(true), bool(true)))
assert(!eq(bool(false), bool(true)))

//integer
assert(eq(integer(0), integer(0)))
assert(!eq(integer(0), integer(1)))
assert(
	eq(
		integer(1_000_000_000_000_000_000_000_000n),
		integer(1_000_000_000_000n * 1_000_000_000_000n)
	)
)
assert(
	eq(
		integer(1_000_000_000_000_000_000_000_000n),
		integer('1000000000000000000000000')
	)
)
assert(
	eq(
		integer(1_000_000_000_000_000_000_000_000n),
		integer('+1000000000000000000000000')
	)
)
assert(
	eq(
		integer(-1_000_000_000_000_000_000_000_000n),
		integer('-1000000000000000000000000')
	)
)

//distinct object
assert(eq(distinctObj('a'), distinctObj('a')))
assert(!eq(distinctObj('a'), distinctObj('b')))

exports.occurs = occurs
exports.distinctObj = distinctObj
exports.unify = unify
exports.bool = bool
exports.call = call
exports.eq = eq
exports.term = term
exports.variable = variable
