'use strict'
const assert = require('assert')

Error.stackTraceLimit = Infinity

var version = '0'

function cktime(deadline) {
	if (deadline && new Date().getTime() >= deadline) throw 'Timeout'
}

function isnumtype(t) {
	switch (t) {
		case 'bigint':
		case 'rat':
		case 'real':
			return true
	}
}

function defaulttype(a, t) {
	switch (a.o) {
		case 'fn':
		case 'var':
			if (!a.type) a.type = t
			break
		case 'call':
			var f = a[0]
			if (!f.type) f.type = [t].concat(a.slice(1).map(type))
			break
	}
}

function quote(q, s) {
	var r = [q]
	for (var i = 0; i < s.length; i++) {
		switch (s[i]) {
			case q:
			case '\\':
				r.push('\\')
				break
		}
		r.push(s[i])
	}
	r.push(q)
	return r.join('')
}

function show(a) {
	console.dir(a, { depth: null })
}

function type(a) {
	switch (typeof a) {
		case 'boolean':
		case 'bigint':
			return typeof a
		case 'string':
			return 'individual'
	}
	if (a.type) return a.type
	switch (a.o) {
		case '&&':
		case '||':
		case '!':
		case '<=>':
		case '==':
		case '<':
		case '<=':
		case 'all':
		case 'exists':
		case 'isint':
		case 'israt':
			return 'boolean'
		case 'toint':
			return 'bigint'
		case 'torat':
			return 'rat'
		case 'toreal':
			return 'real'
		case '+':
		case '-':
		case 'unary-':
		case 'floor':
		case 'ceil':
		case 'trunc':
		case 'round':
		case '/':
		case '*':
		case '%':
		case 'dive':
		case 'divf':
		case 'divt':
		case 'reme':
		case 'remf':
		case 'remt':
			return type(a[0])
		case 'call':
			var t = type(a[0])
			return t[0]
	}
	show(a)
	assert(false)
}

function occurs(a, b, m) {
	if (a === b) return true
	if (m.has(b)) return occurs(a, m.get(b), m)
	if (!Array.isArray(b)) return
	for (var x of b) if (occurs(a, x, m)) return true
}

function unify(a, b, m = new Map()) {
	if (a === b) return m
	if (type(a) !== type(b)) return
	if (a.o === 'var') return unifyvar(a, b, m)
	if (b.o === 'var') return unifyvar(b, a, m)
	if (!Array.isArray(a)) return
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
	return map(a, (b) => freshvars(b, m))
}

function match(a, b, m = new Map()) {
	if (a === b) return m
	if (type(a) !== type(b)) return
	if (a.o === 'var') {
		if (m.has(a)) return eq(m.get(a), b) ? m : null
		m.set(a, b)
		return m
	}
	if (!Array.isArray(a)) return
	if (a.o !== b.o) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length && m; i++) m = match(a[i], b[i], m)
	return m
}

function isomorphic(a, b, m = new Map()) {
	if (a.o !== b.o) return
	// there is some superficial similarity to unification here, but the logical sequence is different
	// in particular, identity of terms is not checked until after variables
	// because two identical variables should not be cleared as isomorphic
	// without checking for bindings
	if (a.o === 'var') {
		if (a.type !== b.type) return
		if (m.has(a) && m.has(b)) {
			if (m.get(a) === b) {
				assert(m.get(b) === a)
				return m
			}
			return
		}
		if (!m.has(a) && !m.has(b)) {
			m.set(a, b)
			m.set(b, a)
			return m
		}
		return
	}
	if (a === b) return m
	if (!Array.isArray(a)) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length && m; i++) m = isomorphic(a[i], b[i], m)
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

function eqn(a) {
	if (a.o === '==') return a
	return mk('==', a, true)
}

function extension(file) {
	var a = file.split('.')
	if (a.length < 2) return ''
	return a.pop()
}

function eq(a, b) {
	if (a === b) return true
	if (!Array.isArray(a)) return
	if (a.o !== b.o) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length; i++) if (!eq(a[i], b[i])) return
	return true
}

function isconst(a) {
	switch (typeof a) {
		case 'bigint':
		case 'string':
			return true
	}
}

function subset(s, t) {
	if (Array.isArray(s)) s = new Set(s)
	if (Array.isArray(t)) t = new Set(t)
	for (var a of s) if (!t.has(a)) return
	return true
}

function eqsets(s, t) {
	if (Array.isArray(s)) s = new Set(s)
	if (Array.isArray(t)) t = new Set(t)
	return s.size === t.size && subset(s, t)
}

function simplify(a, m = new Map()) {
	if (m.has(a)) return simplify(m.get(a), m)
	a = map(a, (b) => simplify(b, m))
	var x = a[0]
	var y = a[1]
	switch (a.o) {
		case 'israt':
			switch (typeof x) {
				case 'bigint':
				case 'rat':
					return true
			}
			break
		case 'isint':
			if (type(x) === 'bigint') return true
			break
		case 'toint':
		case 'round':
		case 'trunc':
		case 'floor':
		case 'ceil':
			if (type(x) === 'bigint') return x
			break
		case 'torat':
			if (type(x) === 'rat') return x
			break
		case 'toreal':
			if (type(x) === 'real') return x
			break
		case '<':
			if (typeof x === 'bigint' && typeof y === 'bigint') return x < y
			break
		case '<=':
			if (typeof x === 'bigint' && typeof y === 'bigint') return x <= y
			break
		case '+':
			if (typeof x === 'bigint' && typeof y === 'bigint') return x + y
			if (x === 0n) return y
			if (y === 0n) return x
			break
		case '*':
			if (typeof x === 'bigint' && typeof y === 'bigint') return x * y
			if (x === 0n) return x
			if (y === 0n) return y
			if (x === 1n) return y
			if (y === 1n) return x
			break
		case 'unary-':
			if (typeof x === 'bigint') return -x
			break
		case '-':
			if (typeof x === 'bigint' && typeof y === 'bigint') return x - y
			if (x === 0n) return mk('unary-', y)
			if (y === 0n) return x
			break
		case '==':
			if (eq(x, y)) return true
			if (isconst(x) && isconst(y)) return false
			break
	}
	return a
}

function map(a, f) {
	if (!Array.isArray(a)) return a
	var r = []
	Object.assign(r, a)
	for (var i = 0; i < r.length; i++) r[i] = f(r[i])
	return r
}

function cartproduct(qs) {
	var js = []
	for (var q of qs) js.push(0)
	var rs = []

	function rec(i) {
		if (i === js.length) {
			var ys = []
			for (i = 0; i < js.length; i++) ys.push(qs[i][js[i]])
			rs.push(ys)
			return
		}
		for (js[i] = 0; js[i] < qs[i].length; js[i]++) rec(i + 1)
	}

	rec(0)
	return rs
}

function mk(o, ...args) {
	var a = Array.from(args)
	a.o = o
	return a
}

function replace(a, m) {
	if (m.has(a)) return replace(m.get(a), m)
	return map(a, (b) => replace(b, m))
}

function getor(m, k, f) {
	if (m.has(k)) return m.get(k)
	var v = f()
	m.set(k, v)
	return v
}

function walk(a, f) {
	f(a)
	if (Array.isArray(a)) for (var b of a) walk(b, f)
}

function test() {
	// default param
	function f(a = []) {
		a.push(1)
		return a
	}

	assert(f().length === 1)
	assert(f().length === 1)

	// getor
	var m = new Map()

	assert(getor(m, 'a', () => 5) === 5)
	assert(m.size === 1)
	assert(m.get('a') === 5)

	assert(getor(m, 'a', () => 5) === 5)
	assert(m.size === 1)
	assert(m.get('a') === 5)

	assert(getor(m, 'b', () => 6) === 6)
	assert(m.size === 2)
	assert(m.get('a') === 5)
	assert(m.get('b') === 6)

	// concat
	function g(...s) {
		return s
	}

	assert(g(1, 2).length === 2)
	assert(g(...[1, 2]).length === 2)
	assert(g(...[1, 2].concat([3, 4])).length === 4)

	// cartesian product
	var qs = []
	var q = null
	q = []
	q.push('a0')
	q.push('a1')
	qs.push(q)
	q = []
	q.push('b0')
	q.push('b1')
	q.push('b2')
	qs.push(q)
	q = []
	q.push('c0')
	q.push('c1')
	q.push('c2')
	q.push('c3')
	qs.push(q)
	var rs = cartproduct(qs)
	var i = 0
	assert(eq(rs[i++], ['a0', 'b0', 'c0']))
	assert(eq(rs[i++], ['a0', 'b0', 'c1']))
	assert(eq(rs[i++], ['a0', 'b0', 'c2']))
	assert(eq(rs[i++], ['a0', 'b0', 'c3']))
	assert(eq(rs[i++], ['a0', 'b1', 'c0']))
	assert(eq(rs[i++], ['a0', 'b1', 'c1']))
	assert(eq(rs[i++], ['a0', 'b1', 'c2']))
	assert(eq(rs[i++], ['a0', 'b1', 'c3']))
	assert(eq(rs[i++], ['a0', 'b2', 'c0']))
	assert(eq(rs[i++], ['a0', 'b2', 'c1']))
	assert(eq(rs[i++], ['a0', 'b2', 'c2']))
	assert(eq(rs[i++], ['a0', 'b2', 'c3']))
	assert(eq(rs[i++], ['a1', 'b0', 'c0']))
	assert(eq(rs[i++], ['a1', 'b0', 'c1']))
	assert(eq(rs[i++], ['a1', 'b0', 'c2']))
	assert(eq(rs[i++], ['a1', 'b0', 'c3']))
	assert(eq(rs[i++], ['a1', 'b1', 'c0']))
	assert(eq(rs[i++], ['a1', 'b1', 'c1']))
	assert(eq(rs[i++], ['a1', 'b1', 'c2']))
	assert(eq(rs[i++], ['a1', 'b1', 'c3']))
	assert(eq(rs[i++], ['a1', 'b2', 'c0']))
	assert(eq(rs[i++], ['a1', 'b2', 'c1']))
	assert(eq(rs[i++], ['a1', 'b2', 'c2']))
	assert(eq(rs[i++], ['a1', 'b2', 'c3']))

	// eqn
	var a = { o: 'fn' }
	assert(eq(eqn(a), mk('==', a, true)))
	assert(eq(eqn(true), mk('==', true, true)))
	assert(eq(eqn(mk('call', a, 1, 2)), mk('==', mk('call', a, 1, 2), true)))
	assert(eq(eqn(mk('==', 1, 2)), mk('==', 1, 2)))

	// map
	assert(
		!eq(
			mk('+', 1, 2).map((a) => a + 10),
			mk('+', 11, 12)
		)
	)
	assert(
		eq(
			map(mk('+', 1, 2), (a) => a + 10),
			mk('+', 11, 12)
		)
	)

	// simplify
	var x = {}
	var y = {}
	var f = {}
	assert(eq(simplify(x), x))
	var m = new Map()
	m.set(x, y)
	assert(eq(simplify(x, m), y))
	assert(eq(simplify(mk('call', f, x, y), m), mk('call', f, y, y)))

	// fn
	var a = { o: 'fn', type: 'individual' }
	var b = { o: 'fn', type: 'individual' }
	assert(eq(a, a))
	assert(!eq(a, b))

	// variable
	var x = { o: 'var', type: 'individual' }
	var y = { o: 'var', type: 'individual' }
	var z = { o: 'var', type: 'individual' }
	assert(!Array.isArray(x))
	assert(eq(x, x))
	assert(eq(y, y))
	assert(!eq(x, y))
	assert(x === x)
	assert(x !== y)
	var xs = new Set()
	xs.add(x)
	assert(xs.has(x))
	assert(!xs.has(y))

	// term
	assert(eq(mk('&&', true, true), mk('&&', true, true)))
	assert(eq(mk('&&', true, true), mk('&&', ...[true, true])))
	assert(!eq(mk('&&', true, true), mk('||', true, true)))
	assert(!eq(mk('&&', true, true), mk('&&', true, false)))
	assert(!eq(mk('&&', true, true), x))

	// arrays
	assert(!eq([true, true], x))
	assert(!eq([true, true], true))

	// call
	var f = { o: 'fn' }
	var g = { o: 'fn' }
	assert(eq(mk('call', f, 1n, 2n), mk('call', f, 1n, 2n)))
	assert(!eq(mk('call', f, 1n, 2n), mk('call', g, 1n, 2n)))
	assert(!eq(mk('call', f, 1n, 2n), mk('call', f, 1n, 3n)))

	// https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
	var f1 = { o: 'fn', type: ['individual', 'individual'] }
	var f2 = { o: 'fn', type: ['individual', 'individual', 'individual'] }
	var g1 = { o: 'fn', type: ['individual', 'individual'] }
	var g2 = { o: 'fn', type: ['individual', 'individual', 'individual'] }
	var m = null

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
	assert(unify(mk('call', f2, a, x), mk('call', f2, a, b), m))
	assert(m.size === 1)
	assert(eq(replace(x, m), b))

	// f and g do not match
	m = new Map()
	assert(!unify(mk('call', f1, a), mk('call', g1, a), m))

	// x and y are aliased
	m = new Map()
	assert(unify(mk('call', f1, x), mk('call', f1, y), m))
	assert(m.size === 1)
	assert(eq(replace(x, m), replace(y, m)))

	// f and g do not match
	m = new Map()
	assert(!unify(mk('call', f1, x), mk('call', g1, y), m))

	// Fails. The f function symbols have different arity
	m = new Map()
	assert(!unify(mk('call', f1, x), mk('call', f2, y, z), m))

	// Unifies y with the term g(x)
	m = new Map()
	assert(unify(mk('call', f1, mk('call', g1, x)), mk('call', f1, y), m))
	assert(m.size === 1)
	assert(eq(replace(y, m), mk('call', g1, x)))

	// Unifies x with constant a, and y with the term g(a)
	m = new Map()
	assert(unify(mk('call', f2, mk('call', g1, x), x), mk('call', f2, y, a), m))
	assert(m.size === 2)
	assert(eq(replace(x, m), a))
	assert(eq(replace(y, m), mk('call', g1, a)))

	// Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
	m = new Map()
	assert(!unify(x, mk('call', f1, x), m))

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
	assert(match(mk('call', f2, a, x), mk('call', f2, a, b), m))
	assert(m.size === 1)
	assert(eq(replace(x, m), b))

	// f and g do not match
	m = new Map()
	assert(!match(mk('call', f1, a), mk('call', g1, a), m))

	// x and y are aliased
	m = new Map()
	assert(match(mk('call', f1, x), mk('call', f1, y), m))
	assert(m.size === 1)
	assert(eq(replace(x, m), replace(y, m)))

	// f and g do not match
	m = new Map()
	assert(!match(mk('call', f1, x), mk('call', g1, y), m))

	// Fails. The f function symbols have different arity
	m = new Map()
	assert(!match(mk('call', f1, x), mk('call', f2, y, z), m))

	// Unifies y with the term g(x)
	// different result for match!
	m = new Map()
	assert(!match(mk('call', f1, mk('call', g1, x)), mk('call', f1, y), m))

	// Unifies x with constant a, and y with the term g(a)
	// different result for match!
	m = new Map()
	assert(!match(mk('call', f2, mk('call', g1, x), x), mk('call', f2, y, a), m))

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

	// freevars
	var s = freevars(mk('call', f2, x, y))
	assert(s.size === 2)
	assert(s.has(x))
	assert(s.has(y))

	var s = freevars(mk('all', [x], mk('call', f2, x, y)))
	assert(s.size === 1)
	assert(s.has(y))

	// freshvars
	var y = freshvars(x)
	assert(y.o === 'var')
	assert(y !== x)

	var b = freshvars(5)
	assert(b === 5)

	var b = freshvars(mk('call', f2, x, y))
	assert(b.o === 'call')
	assert(b[0] === f2)
	var x1 = b[1]
	assert(x1.o === 'var')
	assert(x1 !== x && x1 !== y)
	var y1 = b[2]
	assert(y1.o === 'var')
	assert(y1 !== x && y1 !== y)
	assert(x1 !== y1)

	// type
	assert(type(true) === 'boolean')
	assert(type(9n) === 'bigint')
	assert(type({ o: 'var', type: 'rat' }) === 'rat')
	assert(type(mk('==', 3, 3)) === 'boolean')
	var p2 = { o: 'fn', type: ['boolean', 'individual', 'individual'] }
	assert(type(mk('call', p2, '3', '3')) === 'boolean')

	// quote
	assert(quote('|', 'abc') === '|abc|')
	assert(quote('|', 'ab|c') === '|ab\\|c|')

	// defaulttype
	x = { o: 'var' }
	defaulttype(x, 'real')
	assert(type(x) === 'real')
	defaulttype(x, 'rat')
	assert(type(x) === 'real')

	a = { o: 'fn' }
	defaulttype(a, 'real')
	assert(type(a) === 'real')
	defaulttype(a, 'rat')
	assert(type(a) === 'real')

	f = { o: 'fn' }
	a = mk('call', f, 1n, 2n)
	defaulttype(a, 'real')
	assert(type(a) === 'real')
	assert(eq(type(f), ['real', 'bigint', 'bigint']))
	defaulttype(a, 'rat')
	assert(type(a) === 'real')
	assert(eq(type(f), ['real', 'bigint', 'bigint']))

	// isomorphic
	assert(isomorphic(1n, 1n))
	assert(isomorphic(mk('==', 1n, 1n), mk('==', 1n, 1n)))
	assert(!isomorphic(mk('==', 1n, 1n), mk('==', 1n, 2n)))
	x = { o: 'var', type: 'bigint' }
	y = { o: 'var', type: 'bigint' }
	assert(isomorphic(x, x))
	assert(isomorphic(x, y))
	assert(!isomorphic(x, 1n))
	assert(isomorphic(mk('==', x, x), mk('==', x, x)))
	assert(!isomorphic(mk('==', x, x), mk('==', x, y)))

	// simplify
	assert(simplify(5n) === 5n)
	assert(simplify(x) === x)
	assert(simplify(mk('==', x, x)) === true)
	assert(eq(simplify(mk('==', x, y)), mk('==', x, y)))
	assert(eq(simplify(mk('==', 1000000n, 1000000n)), true))
	assert(eq(simplify(mk('==', 1n, 2n)), false))
	assert(eq(simplify(mk('<', 1n, 2n)), true))
	assert(eq(simplify(mk('+', 1n, 2n)), 3n))
	assert(eq(simplify(mk('+', x, 0n)), x))
	assert(eq(simplify(mk('*', x, 0n)), 0n))
	assert(eq(simplify(mk('*', x, 1n)), x))
	assert(eq(simplify(mk('isint', 2n)), true))
	assert(eq(simplify(mk('toint', 2n)), 2n))

	// subset
	assert(subset(new Set([1, 2, 3]), new Set([1, 2, 3])))
	assert(!subset(new Set([1, 2, 3, 4]), new Set([1, 2, 3])))
	assert(subset(new Set([1, 2, 3]), new Set([1, 2, 3, 4])))
	assert(subset(new Set([2, 3]), new Set([1, 2, 3])))

	// eqsets
	assert(eqsets(new Set([1, 2, 3]), new Set([1, 2, 3])))
	assert(!eqsets(new Set([1, 2, 3, 4]), new Set([1, 2, 3])))
	assert(!eqsets(new Set([1, 2, 3]), new Set([1, 2, 3, 4])))
	assert(!eqsets(new Set([2, 3]), new Set([1, 2, 3])))
	assert(eqsets([1, 2, 3], new Set([1, 2, 3])))
	assert(!eqsets([1, 2, 3, 4], new Set([1, 2, 3])))
	assert(!eqsets(new Set([1, 2, 3]), [1, 2, 3, 4]))
	assert(!eqsets(new Set([2, 3]), [1, 2, 3]))
}

test()

exports.walk = walk
exports.getor = getor
exports.eq = eq
exports.map = map
exports.mk = mk
exports.replace = replace
exports.cartproduct = cartproduct
exports.extension = extension
exports.eqn = eqn
exports.simplify = simplify
exports.unify = unify
exports.match = match
exports.freevars = freevars
exports.freshvars = freshvars
exports.type = type
exports.show = show
exports.quote = quote
exports.defaulttype = defaulttype
exports.isnumtype = isnumtype
exports.version = version
exports.cktime = cktime
exports.isomorphic = isomorphic
exports.subset = subset
exports.eqsets = eqsets
