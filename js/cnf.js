'use strict'
var logic = require('./logic')
var assert = require('assert')

function clause(neg, pos) {
	assert(!logic.isTerm(neg))
	assert(!logic.isTerm(pos))
	neg = logic.term('bag', ...neg)
	pos = logic.term('bag', ...pos)
	var c = [neg, pos]
	c.op = 'clause'
	return c
}

function clause1(a) {
	var neg = []
	var pos = []

	function rec(a) {
		assert(logic.isTerm(a))
		assert(a.op != '&&')
		switch (a.op) {
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
	return clause(neg, pos)
}

function simplifyClause(c, m) {
	;[neg, pos] = c
	neg = neg.map((a) => logic.simplify(a, m))
}

var a = logic.fn('a')
var b = logic.fn('b')
assert(logic.eq(clause([a], [b]), clause([a], [b])))
assert(logic.eq(clause([a], [b]), clause1(logic.term('||', logic.term('!', a), b))))

exports.clause = clause
