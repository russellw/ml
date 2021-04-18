'use strict'
const assert = require('assert')
const logic = require('./logic')
const etc = require('./etc')

function match(c0, d0, c1, d1, m) {
	// empty list means we have matched everything in one polarity
	// note the asymmetry:
	// for c to subsume d, we need to match every c literal
	// but it's okay to have leftover d literals
	if (!c0.length) {
		// try the other polarity
		if (c1) return match(c1, d1, null, null, m)
		// have already matched everything in the other polarity
		return m
	}

	// try matching literals
	for (var ci = 0; ci < c0.length; ci++) {
		// make an equation out of each literal
		// because an equation can be matched either way around
		var ce = logic.eqn(c0[ci])
		// if we successfully match a literal
		// it can be removed from further consideration
		// on this branch of the search tree
		// so make a copy of this list of literals
		// minus the candidate lateral we are trying to match
		var cx = c0.slice()
		cx.splice(ci, 1)
		for (var di = 0; di < d0.length; di++) {
			// same thing with the literals on the other side
			var de = logic.eqn(d0[di])
			var dx = d0.slice()
			dx.splice(di, 1)

			// try orienting equation one way
			var m1 = new Map(m)
			if (logic.match(ce[0], de[0], m1) && logic.match(ce[1], de[1], m1)) {
				// if we successfully match this pair of literals
				// need to continue with the backtracking search
				// to see if these variable assignments also let us match all the other literals
				m1 = match(cx, dx, c1, d1, m1)
				if (m1) return m1
			}

			// and the other way
			var m1 = new Map(m)
			if (logic.match(ce[0], de[1], m1) && logic.match(ce[1], de[0], m1)) {
				m1 = match(cx, dx, c1, d1, m1)
				if (m1) return m1
			}

			// if this pair of literals did not match
			// in either orientation of the respective equations
			// continue to look at all the other possible pairs of literals
		}
	}
}

function subsumes(c, d) {
	console.log('=============================================')
	console.dir(c === d)
	console.dir(c, { depth: null })
	console.dir(d, { depth: null })
	console.log(logic.freevars(c))
	console.log(logic.freevars(d))
	assert(c.length === 2)
	assert(d.length === 2)

	// clauses are assumed to have distinct variable names
	for (x of logic.freevars(c)) assert(!logic.freevars(d).has(x))

	// negative and positive sides need to be matched separately
	// though of course with shared variable assignments
	var [c0, c1] = c
	var [d0, d1] = d

	// it is impossible for a longer clause to subsume a shorter one
	if (c0.length > d0.length || c1.length > d1.length) return

	// fewer literals typically fail faster
	console.trace([c0, d0, c1, d1])
	if (c1.length < c0.length) {
		;[c1, c0] = c
		;[d1, d0] = d
		console.trace([c0, d0, c1, d1])
	}
	console.trace([c0, d0, c1, d1])

	// search for matched literals
	return match(c0, d0, c1, d1, new Map())
}

// test
var a = { type: 'individual' }
var a1 = { type: ['individual', 'individual'] }
var b = { type: 'individual' }
var p = { type: 'bool' }
var p1 = { type: ['bool', 'individual'] }
var p2 = { type: ['bool', 'individual', 'individual'] }
var q = { type: 'bool' }
var q1 = { type: ['bool', 'individual'] }
var x = { o: 'var', name: 'x', type: 'individual' }
var y = { o: 'var', name: 'y', type: 'individual' }
var c, d

// false <= false
assert(x.name !== y.name)
c = [[], []]
d = [[], []]
assert(subsumes(c, d))
assert(subsumes(d, c))

// false <= p
assert(x.name !== y.name)
c = [[], []]
d = [[], [p]]
assert(subsumes(c, d))
assert(!subsumes(d, c))

// p <= p
assert(x.name !== y.name)
c = [[], [p]]
d = [[], [p]]
assert(subsumes(c, d))
assert(subsumes(d, c))

// !p <= !p
assert(x.name !== y.name)
c = [[p], []]
d = [[p], []]
assert(subsumes(c, d))
assert(subsumes(d, c))

// p <= p | p
assert(x.name !== y.name)
c = [[], [p]]
d = [[], [p, p]]
assert(subsumes(c, d))
assert(!subsumes(d, c))

// p !<= !p
assert(x.name !== y.name)
c = [[], [p]]
d = [[p], []]
assert(!subsumes(c, d))
assert(!subsumes(d, c))

// p | q <= q | p
assert(x.name !== y.name)
c = [[], [p, q]]
d = [[], [q, p]]
assert(subsumes(c, d))
assert(subsumes(d, c))

// p | q <= p | q | p
assert(x.name !== y.name)
c = [[], [p, q]]
d = [[], [p, q, p]]
assert(subsumes(c, d))
assert(!subsumes(d, c))

// p(a) | p(b) | q(a) | q(b) | <= p(a) | q(a) | p(b) | q(b)
assert(x.name !== y.name)
c = [[], [etc.mk('call', p, a), etc.mk('call', p, b), etc.mk('call', q, a), etc.mk('call', q, b)]]
d = [[], [etc.mk('call', p, a), etc.mk('call', q, a), etc.mk('call', p, b), etc.mk('call', q, b)]]
assert(subsumes(c, d))
assert(subsumes(d, c))

// p(x,y) <= p(a,b)
assert(x.name !== y.name)
c = [[], [etc.mk('call', p2, x, y)]]
d = [[], [etc.mk('call', p2, a, b)]]
assert(x.name !== y.name)
assert(subsumes(c, d))
assert(x.name !== y.name)
assert(!subsumes(d, c))

// p(x,x) !<= p(a,b)
assert(x.name !== y.name)
c = [[], [etc.mk('call', p2, x, x)]]
d = [[], [etc.mk('call', p2, a, b)]]
assert(!subsumes(c, d))
assert(!subsumes(d, c))

// p(x) <= p(y)
console.dir(x)
console.dir(y)
assert(x.name !== y.name)
c = [[], [etc.mk('call', p1, x)]]
d = [[], [etc.mk('call', p1, y)]]
assert(subsumes(c, d))
assert(subsumes(d, c))

// p(x) | p(a(x)) | p(a(a(x))) <= p(y) | p(a(y)) | p(a(a(y)))
negative.clear()
positive.clear()
positive.add(List.of(p1, x))
positive.add(List.of(p1, List.of(a1, x)))
positive.add(List.of(p1, List.of(a1, List.of(a1, x))))
c = new Clause(negative, positive, Inference.AXIOM)
negative.clear()
positive.clear()
positive.add(List.of(p1, y))
positive.add(List.of(p1, List.of(a1, y)))
positive.add(List.of(p1, List.of(a1, List.of(a1, y))))
d = new Clause(negative, positive, Inference.AXIOM)
assert(subsumes(c, d))
assert(subsumes(d, c))

// p(x) | p(a) <= p(a) | p(b)
negative.clear()
positive.clear()
positive.add(List.of(p1, x))
positive.add(List.of(p1, a))
c = new Clause(negative, positive, Inference.AXIOM)
negative.clear()
positive.clear()
positive.add(List.of(p1, a))
positive.add(List.of(p1, b))
d = new Clause(negative, positive, Inference.AXIOM)
assert(subsumes(c, d))
assert(!subsumes(d, c))

// p(x) | p(a(x)) <= p(a(y)) | p(y)
negative.clear()
positive.clear()
positive.add(List.of(p1, x))
positive.add(List.of(p1, List.of(a1, x)))
c = new Clause(negative, positive, Inference.AXIOM)
negative.clear()
positive.clear()
positive.add(List.of(p1, List.of(a1, y)))
positive.add(List.of(p1, y))
d = new Clause(negative, positive, Inference.AXIOM)
assert(subsumes(c, d))
assert(subsumes(d, c))

// p(x) | p(a(x)) | p(a(a(x))) <= p(a(a(y))) | p(a(y)) | p(y)
negative.clear()
positive.clear()
positive.add(List.of(p1, x))
positive.add(List.of(p1, List.of(a1, x)))
positive.add(List.of(p1, List.of(a1, List.of(a1, x))))
c = new Clause(negative, positive, Inference.AXIOM)
negative.clear()
positive.clear()
positive.add(List.of(p1, List.of(a1, List.of(a1, y))))
positive.add(List.of(p1, List.of(a1, y)))
positive.add(List.of(p1, y))
d = new Clause(negative, positive, Inference.AXIOM)
assert(subsumes(c, d))
assert(subsumes(d, c))

// (a = x) <= (a = b)
negative.clear()
positive.clear()
positive.add(Equality.of(a, x))
c = new Clause(negative, positive, Inference.AXIOM)
negative.clear()
positive.clear()
positive.add(Equality.of(a, b))
d = new Clause(negative, positive, Inference.AXIOM)
assert(subsumes(c, d))
assert(!subsumes(d, c))

// (x = a) <= (a = b)
negative.clear()
positive.clear()
positive.add(Equality.of(x, a))
c = new Clause(negative, positive, Inference.AXIOM)
negative.clear()
positive.clear()
positive.add(Equality.of(a, b))
d = new Clause(negative, positive, Inference.AXIOM)
assert(subsumes(c, d))
assert(!subsumes(d, c))

// !p(y) | !p(x) | q(x) <= !p(a) | !p(b) | q(b)
negative.clear()
negative.add(List.of(p1, y))
negative.add(List.of(p1, x))
positive.clear()
positive.add(List.of(q1, x))
c = new Clause(negative, positive, Inference.AXIOM)
negative.clear()
negative.add(List.of(p1, a))
negative.add(List.of(p1, b))
positive.clear()
positive.add(List.of(q1, b))
d = new Clause(negative, positive, Inference.AXIOM)
assert(subsumes(c, d))
assert(!subsumes(d, c))

// !p(x) | !p(y) | q(x) <= !p(a) | !p(b) | q(b)
negative.clear()
negative.add(List.of(p1, x))
negative.add(List.of(p1, y))
positive.clear()
positive.add(List.of(q1, x))
c = new Clause(negative, positive, Inference.AXIOM)
negative.clear()
negative.add(List.of(p1, a))
negative.add(List.of(p1, b))
positive.clear()
positive.add(List.of(q1, b))
d = new Clause(negative, positive, Inference.AXIOM)
assert(subsumes(c, d))
assert(!subsumes(d, c))

// p(x,a(x)) !<= p(a(y),a(y))
negative.clear()
positive.clear()
positive.add(List.of(p2, x, List.of(a1, x)))
c = new Clause(negative, positive, Inference.AXIOM)
negative.clear()
positive.clear()
positive.add(List.of(p2, List.of(a1, y), List.of(a1, y)))
d = new Clause(negative, positive, Inference.AXIOM)
assert(!subsumes(c, d))
assert(!subsumes(d, c))

// exports
exports.subsumes = subsumes
