'use strict'
const etc = require('./etc')
const assert = require('assert')

function simplify(c, m = new Map()) {
	var [neg, pos] = c

	// simplify
	neg = neg.map((a) => etc.simplify(a, m))
	pos = pos.map((a) => etc.simplify(a, m))

	// filter out redundancy
	neg = neg.filter((a) => a !== true)
	pos = pos.filter((a) => a !== false)

	// tautology?
	for (var a of neg) if (a === false) return [[], [true]]
	for (var a of pos) if (a === true) return [[], [true]]
	for (var a of neg) for (var b of pos) if (etc.eq(a, b)) return [[], [true]]

	// make new clause
	c = [neg, pos]
	ckclause(c)
	return c
}

function all(env, a) {
	env = new Map(env)
	for (var x of a[0]) env.set(x, { o: 'var', type: x.type })
	return env
}

function exists(env, a) {
	var params = []
	for (var [k, v] of env.entries()) if (v.o === 'var' && etc.freevars(a[1]).has(k)) params.push(v)
	env = new Map(env)
	for (var x of a[0]) env.set(x, skolem(x.type, params))
	return env
}

function skolem(rt, params) {
	var sk = { o: 'fn', type: rt }
	if (!params.length) return sk
	sk.type = [rt].concat(params.map(etc.type))
	return etc.mk('call', ...[sk].concat(params))
}

function convert(c, clauses) {
	// most of the work is done in conversion to negation normal form
	// the logic of which depends on whether the caller wants a negative or positive version of the formula
	// or both, if the caller was an equivalence
	function nnfneg(env, a) {
		if (typeof a === 'boolean') return !a
		switch (a.o) {
			case 'all':
				var body = a[1]
				return nnfneg(exists(env, a), body)
			case 'exists':
				var body = a[1]
				return nnfneg(all(env, a), body)
			case '!':
				return nnfpos(env, a[0])
			case '=>':
				return nnfneg(env, etc.mk('||', etc.mk('!', a[0]), a[1]))
			case '&&':
				return etc.mk('||', ...a.map((b) => nnfneg(env, b)))
			case '||':
				return etc.mk('&&', ...a.map((b) => nnfneg(env, b)))
			case 'var':
				assert(env.has(a))
				return env.get(a)
			case '<=>':
				var x = nnfboth(env, a[0])
				var y = nnfboth(env, a[1])
				return etc.mk('&&', etc.mk('||', x[0], y[0]), etc.mk('||', x[1], y[1]))
		}
		return etc.mk(
			'!',
			etc.map(a, (b) => nnfpos(env, b))
		)
	}

	function nnfpos(env, a) {
		switch (a.o) {
			case 'all':
				var body = a[1]
				return nnfpos(all(env, a), body)
			case 'exists':
				var body = a[1]
				return nnfpos(exists(env, a), body)
			case '!':
				return nnfneg(env, a[0])
			case '=>':
				return nnfpos(env, etc.mk('||', etc.mk('!', a[0]), a[1]))
			case '&&':
				return etc.mk('&&', ...a.map((b) => nnfpos(env, b)))
			case '||':
				return etc.mk('||', ...a.map((b) => nnfpos(env, b)))
			case 'var':
				assert(env.has(a))
				return env.get(a)
			case '<=>':
				var x = nnfboth(env, a[0])
				var y = nnfboth(env, a[1])
				return etc.mk('&&', etc.mk('||', x[0], y[1]), etc.mk('||', x[1], y[0]))
		}
		return etc.map(a, (b) => nnfpos(env, b))
	}

	// rename a formula that will be used as an equivalence argument
	// that means both negative and positive versions are needed
	// so the renaming needs implications in both directions
	// that means we have to construct an equivalence involving the formula
	// on the face of it, that puts us back where we started
	// but the saving grace is that the definitional equivalence is a separate thing outside the original context
	// which means it does not need to use the original environment
	// which means all free variables of the formula are universally quantified
	// and the positive and negative versions do not differ in the meanings they assign to the variables
	function renameboth(a) {
		// treat all free variables of the formula as universally quantified
		// with the exception that we don't need to bother renaming them
		// the actual handling of universal quantifiers involves renaming the variables
		// in case the same names are reused in different parts of an overall formula
		// but here we are dealing with exactly one layer
		// and if these variable names are reused in quantifiers in subformulas
		// then those occurrences will be renamed
		// unless those subformulas are themselves renamed
		// in which case they will end up in different clauses
		// so the variables will still not overlap
		var xs = Array.from(etc.freevars(a))
		var env = new Map()
		for (var x of xs) env.set(x, x)

		// need both positive and negative versions of the formula
		var a2 = nnfboth(env, a)

		// b is defined as being equal to a
		// it needs to take the free variables of a as parameters
		var b = skolem('boolean', xs)

		// b implies and is implied by a
		// by construction, given that a was converted to NNF, the expanded equivalence is still so
		// because NOT is placed only over an atomic term
		// but in general the expanded equivalence is not in CNF
		// because OR is placed over converted subformulas which could contain AND
		// so we continue the conversion process from the point where AND needs to rise over OR
		// and then to generate clauses to define the new symbol
		convertrise(etc.mk('&&', etc.mk('||', etc.mk('!', b), a2[1]), etc.mk('||', b, a2[0])))

		// return the new name by which the caller shall now know the formula
		return b
	}

	function nnfboth(env, a) {
		switch (a) {
			case false:
				return [true, false]
			case true:
				return [false, true]
		}
		switch (a.o) {
			case 'all':
				var body = a[1]
				return [nnfneg(exists(env, a), body), nnfpos(all(env, a), body)]
			case 'exists':
				var body = a[1]
				return [nnfneg(all(env, a), body), nnfpos(exists(env, a), body)]
			case '!':
				a = nnfboth(env, a[0])
				return [a[1], a[0]]
			case '=>':
				return nnfboth(env, etc.mk('||', etc.mk('!', a[0]), a[1]))
			case '&&':
				var a2 = a.map((b) => nnfboth(env, b))
				return [etc.mk('||', ...a2.map((b) => b[0])), etc.mk('&&', ...a2.map((b) => b[1]))]
			case '||':
				var a2 = a.map((b) => nnfboth(env, b))
				return [etc.mk('&&', ...a2.map((b) => b[0])), etc.mk('||', ...a2.map((b) => b[1]))]
			case 'var':
				assert(env.has(a))
				return env.get(a)
			case '<=>':
				var x = nnfboth(env, a[0])
				var y = nnfboth(env, a[1])
				return [
					etc.mk('&&', etc.mk('||', x[0], y[0]), etc.mk('||', x[1], y[1])),
					etc.mk('&&', etc.mk('||', x[0], y[1]), etc.mk('||', x[1], y[0])),
				]
		}
		a = etc.map(a, (b) => nnfpos(env, b))
		return [etc.mk('!', a), a]
	}

	// make AND rise to the top
	function rise(a) {
		a = etc.map(a, rise)
		if (a.o !== '||') return a

		// now we know this term is an OR
		// its arguments can be taken without loss of generality as ANDs
		var ands = []
		for (var b of a) {
			var and = []
			flatten('&&', b, and)
			ands.push(and)
		}

		// OR distributes over AND by Cartesian product
		a = etc.cartproduct(ands)
		for (var b of a) b.o = '||'
		a.o = '&&'
		return a
	}

	var a = c[0]
	a = nnfpos(new Map(), a)
	a = rise(a)

	// now we have a term in CNF
	// need to convert it to actual clauses
	var ors = []
	flatten('&&', a, ors)
	for (var b of ors) {
		var d = cterm(b)
		d.how = 'cnf'
		d.from = [c]
		ckclause(d)
		clauses.push(d)
	}
}

function ckclauses(cs) {
	for (var c of cs) ckclause(c)
}

function ckclause(c) {
	assert(!c.file || typeof c.file === 'string')
	assert(!c.from || Array.isArray(c.from))
	assert(!c.how || typeof c.how === 'string')
	assert(!c.name || typeof c.name === 'string')
	assert(!c.o)
	assert(Array.isArray(c))
	assert(c.length === 2)
	for (var L of c) {
		assert(!L.o)
		assert(Array.isArray(L))
		etc.walk(L, (a) => {
			switch (a.o) {
				case '&&':
				case '||':
				case '!':
				case '=>':
				case '<=>':
				case '!=':
				case 'all':
				case 'exists':
					etc.show(c)
					assert(false)
			}
		})
	}
}

function flatten(o, a, r) {
	if (a.o === o) {
		for (var b of a) flatten(o, b, r)
		return
	}
	r.push(a)
}

function cterm(a) {
	var neg = []
	var pos = []

	function rec(a) {
		switch (a.o) {
			case '&&':
			case '=>':
			case '<=>':
			case '!=':
			case 'all':
			case 'exists':
				assert(false)
			case '||':
				for (var b of a) rec(b)
				return
			case '!':
				neg.push(a[0])
				return
		}
		pos.push(a)
	}

	rec(a)
	return [neg, pos]
}

function test() {
	// clause
	var a = { o: 'fn' }
	var b = { o: 'fn' }
	assert(etc.eq(simplify([[a], [b]]), [[a], [b]]))
	assert(etc.eq(simplify([[a], [b]]), cterm(etc.mk('||', etc.mk('!', a), b))))
	assert(etc.eq(simplify([[a], [false]]), [[a], []]))
	assert(etc.eq(simplify([[a], [true]]), [[], [true]]))
	assert(etc.eq(simplify([[a], [a]]), [[], [true]]))
	var m = new Map()
	m.set(b, false)
	assert(etc.eq(simplify([[a], [b]], m), simplify([[a], []])))

	// flatten
	var r = []
	flatten('+', etc.mk('+', etc.mk('+', 1, 2), 3), r)
	assert(etc.eq(r, [1, 2, 3]))

	var r = []
	flatten('+', 4, r)
	assert(etc.eq(r, [4]))

	// convert
	var cs = []
	convert([true], cs)
	assert(cs.length === 1)
	assert(etc.eq(cs[0], [[], [true]]))

	var cs = []
	convert([false], cs)
	assert(cs.length === 1)
	assert(etc.eq(simplify(cs[0]), [[], []]))

	var cs = []
	convert([a], cs)
	assert(cs.length === 1)
	assert(etc.eq(cs[0], [[], [a]]))

	var cs = []
	convert([etc.mk('!', a)], cs)
	assert(cs.length === 1)
	assert(etc.eq(cs[0], [[a], []]))

	var cs = []
	convert([etc.mk('!', etc.mk('!', a))], cs)
	assert(cs.length === 1)
	assert(etc.eq(cs[0], [[], [a]]))

	var cs = []
	convert([etc.mk('=>', a, b)], cs)
	assert(cs.length === 1)
	assert(etc.eq(cs[0], [[a], [b]]))

	var cs = []
	convert([etc.mk('||', a, b)], cs)
	assert(cs.length === 1)
	assert(etc.eq(cs[0], [[], [a, b]]))

	var cs = []
	convert([etc.mk('&&', a, b)], cs)
	assert(cs.length === 2)
	assert(etc.eq(cs[0], [[], [a]]))
	assert(etc.eq(cs[1], [[], [b]]))

	var a1 = { o: 'fn' }
	var b1 = { o: 'fn' }
	var a2 = { o: 'fn' }
	var b2 = { o: 'fn' }

	var cs = []
	convert([etc.mk('||', a, b, a1, b1)], cs)
	assert(cs.length === 1)
	assert(etc.eq(cs[0], [[], [a, b, a1, b1]]))

	var cs = []
	convert([etc.mk('||', etc.mk('||', a1, b1), etc.mk('||', a2, b2))], cs)
	assert(cs.length === 1)
	assert(etc.eq(cs[0], [[], [a1, b1, a2, b2]]))

	var cs = []
	convert([etc.mk('&&', etc.mk('||', a1, b1), etc.mk('||', a2, b2))], cs)
	assert(cs.length === 2)
	assert(etc.eq(cs[0], [[], [a1, b1]]))
	assert(etc.eq(cs[1], [[], [a2, b2]]))

	var cs = []
	convert([etc.mk('||', a, etc.mk('&&', b1, b2))], cs)
	assert(cs.length === 2)
	assert(etc.eq(cs[0], [[], [a, b1]]))
	assert(etc.eq(cs[1], [[], [a, b2]]))

	var x = { o: 'var', type: 'individual' }
	var y = { o: 'var', type: 'individual' }
	var z = { o: 'var', type: 'individual' }
	var f1 = { o: 'fn', type: ['boolean', 'individual'] }
	var f2 = { o: 'fn', type: ['boolean', 'individual', 'individual'] }
	var g1 = { o: 'fn', type: ['boolean', 'individual'] }
	var g2 = { o: 'fn', type: ['boolean', 'individual', 'individual'] }
	var h = { o: 'fn', type: 'individual' }

	assert(etc.match(etc.mk('call', f1, x), etc.mk('call', f1, h)))
	assert(!etc.match(etc.mk('call', f1, h), etc.mk('call', f1, x)))

	function isomorphic1(a, b, m = new Map()) {
		if (a.length !== b.length) return
		for (var i = 0; i < a.length && m; i++) {
			assert(etc.type(a[i]))
			assert(etc.type(b[i]))
			m = etc.match(a[i], b[i], m)
		}
		return m
	}

	function isomorphic(c, d) {
		var m = isomorphic1(c[0], d[0])
		assert(m && isomorphic1(c[1], d[1], m))

		var m = isomorphic1(d[0], c[0])
		assert(m && isomorphic1(d[1], c[1], m))
		return m
	}

	var cs = []
	convert([etc.mk('all', [x], etc.mk('call', f1, x))], cs)
	assert(cs.length === 1)
	assert(isomorphic(cs[0], [[], [etc.mk('call', f1, x)]]))

	var cs = []
	convert([etc.mk('all', [x, y], etc.mk('call', f2, x, y))], cs)
	assert(cs.length === 1)
	assert(isomorphic(cs[0], [[], [etc.mk('call', f2, x, y)]]))

	var cs = []
	convert([etc.mk('all', [x], etc.mk('all', [y], etc.mk('call', f2, x, y)))], cs)
	assert(cs.length === 1)
	assert(isomorphic(cs[0], [[], [etc.mk('call', f2, x, y)]]))

	var cs = []
	convert([etc.mk('exists', [x], etc.mk('call', f1, x))], cs)
	assert(cs.length === 1)
	var m = etc.match(etc.mk('call', f1, x), cs[0][1][0])
	assert(m)
	assert(m.size === 1)
	assert(!Array.isArray(m.get(x)))
	assert(m.get(x).o === 'fn')

	var cs = []
	convert([etc.mk('all', [x], etc.mk('exists', [y], etc.mk('call', f2, x, y)))], cs)
	assert(cs.length === 1)
	var m = etc.match(etc.mk('call', f2, x, y), cs[0][1][0])
	assert(m)
	assert(m.size === 2)
	assert(!Array.isArray(m.get(x)))
	assert(m.get(x).o === 'var')
	assert(Array.isArray(m.get(y)))
	assert(m.get(y).o === 'call')
	assert(m.get(y).length === 2)

	var cs = []
	convert([etc.mk('all', [x], etc.mk('exists', [y], etc.mk('call', f1, y)))], cs)
	assert(cs.length === 1)
	var m = etc.match(etc.mk('call', f1, y), cs[0][1][0])
	assert(m)
	assert(m.size === 1)
	assert(!Array.isArray(m.get(y)))
	assert(m.get(y).o === 'fn')

	var cs = []
	convert([etc.mk('<=>', a, b)], cs)
	assert(cs.length === 2)
	assert(etc.eq(cs[0], [[a], [b]]))
	assert(etc.eq(cs[1], [[b], [a]]))

	function sat(clauses) {
		var atoms = new Set()
		for (var c of clauses) for (var p of c) for (var a of p) atoms.add(a)
		atoms = Array.from(atoms)

		function rec(i, m) {
			var cs = clauses.map((c) => simplify(c, m))
			for (var c of cs) if (etc.eq(c, [[], []])) return
			cs = cs.filter((c) => !etc.eq(c, [[], [true]]))
			if (!cs.length) return m

			assert(i < atoms.length)
			var a = atoms[i++]
			var m1 = new Map(m)
			m1.set(a, false)
			if (rec(i, m1)) return true
			m.set(a, true)
			return rec(i, m)
		}

		return rec(0, new Map())
	}

	var cs = []
	convert([etc.mk('&&', a, b)], cs)
	assert(sat(cs))

	var cs = []
	convert([etc.mk('&&', a, a)], cs)
	assert(sat(cs))

	var cs = []
	convert([etc.mk('&&', a, etc.mk('!', a))], cs)
	assert(!sat(cs))

	function thm(a) {
		var cs = []
		convert([a], cs)
		assert(sat(cs))

		var cs = []
		convert([etc.mk('!', a)], cs)
		assert(!sat(cs))
	}

	thm(true)
	thm(etc.mk('=>', false, a))
	thm(etc.mk('=>', a, a))
	thm(etc.mk('&&', true, true, true))
	thm(etc.mk('||', false, false, true))
	thm(etc.mk('<=>', a, a))

	var p1 = { o: 'fn', name: 'p1' }
	var p2 = { o: 'fn', name: 'p2' }
	var p3 = { o: 'fn', name: 'p3' }

	thm(etc.mk('<=>', p1, etc.mk('<=>', p2, etc.mk('<=>', p1, p2))))
	// thm(etc.mk('<=>', p1, etc.mk('<=>', p2, etc.mk('<=>', p3, etc.mk('<=>', p1, etc.mk('<=>', p2, p3))))))
}

test()

exports.convert = convert
exports.simplify = simplify
exports.ckclause = ckclause
exports.ckclauses = ckclauses
