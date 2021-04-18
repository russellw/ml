'use strict'
const assert = require('assert')
const logic = require('./logic')

function match(c0, d0, c1, d1, m) {
	// empty list means we have matched everything in one polarity
	// note the asymmetry
	// for c to subsume d, we need to match every c literal
	// but it's okay to have leftover d literals
	if (!c.length) {
		// try the other polarity
		if (c1) return match(c1, d1, null, null, m)
		// have already matched everything in the other polarity
		return m
	}

	// try matching literals
	for (var ci = 0; ci < c.length; ci++) {
		// make an equation out of each literal
		// because an equation can be matched either way around
		var ce = logic.eqn(c[ci])
		// if we successfully match a literal
		// it can be removed from further consideration
		// on this branch of the search tree
		// so make a copy of this list of literals
		// minus the candidate lateral we are trying to match
		var cx = c.splice(ci, 1)
		for (var di = 0; di < d.length; di++) {
			// same thing with the literals on the other side
			var de = logic.eqn(d[di])
			var dx = d.splice(di, 1)

			// try orienting equation one way
			var m1 = new Map(m)
			if (logic.match(ce[0], de[0], m1) && logic.match(ce[1], de[1], m1)) {
				// if we successfully match this pair of literals
				// need to continue with the backtracking search
				// to see if these variable assignments also let us match all the other literals
				m1 = match(c0, d0, c1, d1, m1)
				if (m1) return m1
			}

			// and the other way
			var m1 = new Map(m)
			if (logic.match(ce[0], de[1], m1) && logic.match(ce[1], de[0], m1)) {
				m1 = match(c0, d0, c1, d1, m1)
				if (m1) return m1
			}

			// if this pair of literals did not match
			// in either orientation of the respective equations
			// continue to look at all the other possible pairs of literals
		}
	}
}

function subsumes(c, d) {}
