'use strict'
var logic = require('./logic')
var assert = require('assert')

function clause(neg, pos) {
	neg = logic.term('bag', ...neg)
	neg.op = 'bag'

	pos = pos.slice()
	pos.op = 'bag'

	var c = [neg, pos]
	c.op = 'clause'
	return c
}

function clause1(a) {
	assert(logic.isTerm(a))
}

exports.clause = clause
exports.clause1 = clause1
