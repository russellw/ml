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

function sat(clauses) {
	var passive = priorityq.mk(size)
	for (var c of clauses) priorityq.push(passive, c)
	var active = []

	function clause(neg, pos) {
		var c = cnf.clause(neg, pos)
		if (etc.eq(c, [[], [true]])) return
		priorityq.push(passive, c)
	}

	for (;;) {
		var g = priorityq.pop(passive)
		if (!g) return true
	}
}

function test() {
	assert(size(5) === 1)
	assert(size(etc.mk('==', etc.mk('unary-', 10), etc.mk('+', 11, 12))), 3)
}

test()

exports.sat = sat
