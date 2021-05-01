'use strict'
const assert = require('assert')
const etc = require('./etc')

// one clause subsumes another if there exists a variable substitution
// that makes the first clause a sub-multiset of the second
// multiset not set because otherwise a clause could subsume its own factors
// which would break completeness of the superposition calculus
function subsumes(c, d) {
	assert(c.length === 2)
	assert(d.length === 2)

	// clauses are assumed to have distinct variable names
	for (var x of etc.freevars(c)) assert(!etc.freevars(d).has(x))

	// negative and positive sides need to be matched separately
	// though of course with shared variable assignments
	var [c0, c1] = c
	var [d0, d1] = d

	// it is impossible for a longer clause to subsume a shorter one
	if (c0.length > d0.length || c1.length > d1.length) return

	// fewer literals are likely to fail faster
	// so if there are fewer positive literals than negative
	// swap them around and try the positive side first
	if (c1.length < c0.length) {
		;[c1, c0] = c
		;[d1, d0] = d
	}

	// worst-case time is exponential
	// so give up if taking too long
	var steps = 1000

	function match(c0, d0, c1, d1, m) {
		if (!steps) throw 'break'
		steps--

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
			var ce = etc.eqn(c0[ci])
			// if we successfully match a literal
			// it can be removed from further consideration
			// on this branch of the search tree
			// so make a copy of this list of literals
			// minus the candidate lateral we are trying to match
			var cx = [...c0]
			cx.splice(ci, 1)
			for (var di = 0; di < d0.length; di++) {
				// same thing with the literals on the other side
				var de = etc.eqn(d0[di])
				var dx = [...d0]
				dx.splice(di, 1)

				// try orienting equation one way
				var m1 = new Map(m)
				if (etc.match(ce[0], de[0], m1) && etc.match(ce[1], de[1], m1)) {
					// if we successfully match this pair of literals
					// need to continue with the backtracking search
					// to see if these variable assignments also let us match all the other literals
					m1 = match(cx, dx, c1, d1, m1)
					if (m1) return m1
				}

				// and the other way
				var m1 = new Map(m)
				if (etc.match(ce[0], de[1], m1) && etc.match(ce[1], de[0], m1)) {
					m1 = match(cx, dx, c1, d1, m1)
					if (m1) return m1
				}

				// if this pair of literals did not match
				// in either orientation of the respective equations
				// continue to look at all the other possible pairs of literals
			}
		}
	}

	// search for matched literals
	try {
		return match(c0, d0, c1, d1, new Map())
	} catch (e) {
		// if we failed to find proof of subsumption within allocated time
		// completeness requires the conservative assumption of no subsumption
		if (e === 'break') return
		throw e
	}
}

function test() {
	var a = { o: 'fn', type: 'individual' }
	var a1 = { o: 'fn', type: ['individual', 'individual'] }
	var b = { o: 'fn', type: 'individual' }
	var p = { o: 'fn', type: 'boolean' }
	var p1 = { o: 'fn', type: ['boolean', 'individual'] }
	var p2 = { o: 'fn', type: ['boolean', 'individual', 'individual'] }
	var q = { o: 'fn', type: 'boolean' }
	var q1 = { o: 'fn', type: ['boolean', 'individual'] }
	var x = { o: 'var', name: 'x', type: 'individual' }
	var y = { o: 'var', name: 'y', type: 'individual' }
	var c, d

	// false <= false
	c = [[], []]
	d = [[], []]
	assert(subsumes(c, d))
	assert(subsumes(d, c))

	// false <= p
	c = [[], []]
	d = [[], [p]]
	assert(subsumes(c, d))
	assert(!subsumes(d, c))

	// p <= p
	c = [[], [p]]
	d = [[], [p]]
	assert(subsumes(c, d))
	assert(subsumes(d, c))

	// !p <= !p
	c = [[p], []]
	d = [[p], []]
	assert(subsumes(c, d))
	assert(subsumes(d, c))

	// p <= p | p
	c = [[], [p]]
	d = [[], [p, p]]
	assert(subsumes(c, d))
	assert(!subsumes(d, c))

	// p !<= !p
	c = [[], [p]]
	d = [[p], []]
	assert(!subsumes(c, d))
	assert(!subsumes(d, c))

	// p | q <= q | p
	c = [[], [p, q]]
	d = [[], [q, p]]
	assert(subsumes(c, d))
	assert(subsumes(d, c))

	// p | q <= p | q | p
	c = [[], [p, q]]
	d = [[], [p, q, p]]
	assert(subsumes(c, d))
	assert(!subsumes(d, c))

	// p(a) | p(b) | q(a) | q(b) | <= p(a) | q(a) | p(b) | q(b)
	c = [[], [etc.mk('call', p, a), etc.mk('call', p, b), etc.mk('call', q, a), etc.mk('call', q, b)]]
	d = [[], [etc.mk('call', p, a), etc.mk('call', q, a), etc.mk('call', p, b), etc.mk('call', q, b)]]
	assert(subsumes(c, d))
	assert(subsumes(d, c))

	// p(x,y) <= p(a,b)
	c = [[], [etc.mk('call', p2, x, y)]]
	d = [[], [etc.mk('call', p2, a, b)]]
	assert(subsumes(c, d))
	assert(!subsumes(d, c))

	// p(x,x) !<= p(a,b)
	c = [[], [etc.mk('call', p2, x, x)]]
	d = [[], [etc.mk('call', p2, a, b)]]
	assert(!subsumes(c, d))
	assert(!subsumes(d, c))

	// p(x) <= p(y)
	c = [[], [etc.mk('call', p1, x)]]
	d = [[], [etc.mk('call', p1, y)]]
	assert(subsumes(c, d))
	assert(subsumes(d, c))

	// p(x) | p(a(x)) | p(a(a(x))) <= p(y) | p(a(y)) | p(a(a(y)))
	c = [
		[],
		[
			etc.mk('call', p1, x),
			etc.mk('call', p1, etc.mk('call', a1, x)),
			etc.mk('call', p1, etc.mk('call', a1, etc.mk('call', a1, x))),
		],
	]
	d = [
		[],
		[
			etc.mk('call', p1, y),
			etc.mk('call', p1, etc.mk('call', a1, y)),
			etc.mk('call', p1, etc.mk('call', a1, etc.mk('call', a1, y))),
		],
	]
	assert(subsumes(c, d))
	assert(subsumes(d, c))

	// p(x) | p(a) <= p(a) | p(b)
	c = [[], [etc.mk('call', p1, x), etc.mk('call', p1, a)]]
	d = [[], [etc.mk('call', p1, a), etc.mk('call', p1, b)]]
	assert(subsumes(c, d))
	assert(!subsumes(d, c))

	// p(x) | p(a(x)) <= p(a(y)) | p(y)
	c = [[], [etc.mk('call', p1, x), etc.mk('call', p1, etc.mk('call', a1, x))]]
	d = [[], [etc.mk('call', p1, etc.mk('call', a1, y)), etc.mk('call', p1, y)]]
	assert(subsumes(c, d))
	assert(subsumes(d, c))

	// p(x) | p(a(x)) | p(a(a(x))) <= p(a(a(y))) | p(a(y)) | p(y)
	c = [
		[],
		[
			etc.mk('call', p1, x),
			etc.mk('call', p1, etc.mk('call', a1, x)),
			etc.mk('call', p1, etc.mk('call', a1, etc.mk('call', a1, x))),
		],
	]
	d = [
		[],
		[
			etc.mk('call', p1, etc.mk('call', a1, etc.mk('call', a1, y))),
			etc.mk('call', p1, etc.mk('call', a1, y)),
			etc.mk('call', p1, y),
		],
	]
	assert(subsumes(c, d))
	assert(subsumes(d, c))

	// (a = x) <= (a = b)
	c = [[], [etc.mk('==', a, x)]]
	d = [[], [etc.mk('==', a, b)]]
	assert(subsumes(c, d))
	assert(!subsumes(d, c))

	// (x = a) <= (a = b)
	c = [[], [etc.mk('==', x, a)]]
	d = [[], [etc.mk('==', a, b)]]
	assert(subsumes(c, d))
	assert(!subsumes(d, c))

	// !p(y) | !p(x) | q(x) <= !p(a) | !p(b) | q(b)
	c = [[etc.mk('call', p1, y), etc.mk('call', p1, x)], [etc.mk('call', q1, x)]]
	d = [[etc.mk('call', p1, a), etc.mk('call', p1, b)], [etc.mk('call', q1, b)]]
	assert(subsumes(c, d))
	assert(!subsumes(d, c))

	// !p(x) | !p(y) | q(x) <= !p(a) | !p(b) | q(b)
	c = [[etc.mk('call', p1, x), etc.mk('call', p1, y)], [etc.mk('call', q1, x)]]
	d = [[etc.mk('call', p1, a), etc.mk('call', p1, b)], [etc.mk('call', q1, b)]]
	assert(subsumes(c, d))
	assert(!subsumes(d, c))

	// p(x,a(x)) !<= p(a(y),a(y))
	c = [[], [etc.mk('call', p2, x, etc.mk('call', a1, x))]]
	d = [[], [etc.mk('call', p2, etc.mk('call', a1, y), y)]]
	assert(!subsumes(c, d))
	assert(!subsumes(d, c))
}

test()

exports.subsumes = subsumes
