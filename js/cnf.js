'use strict'
var logic = require('./logic')
var assert = require('assert')

function clause(neg, pos) {
	neg = logic.term('bag', ...neg)
	pos = logic.term('bag', ...pos)
	var c = [neg, pos]
	c.op = 'clause'
	return c
}

var falseClause = [[], []]
var trueClause = [[], [true]]

//var falseClauses =

function simplifyClause(c, m = new Map()) {
	var [neg, pos] = c

	//simplify
	neg = neg.map((a) => logic.simplify(a, m))
	pos = pos.map((a) => logic.simplify(a, m))

	//tautology?
	for (var a in neg) if (a === false) return trueClause
	for (var a in pos) if (a === true) return trueClause

	//filter out redundancy
	neg = neg.filter((a) => a !== true)
	pos = pos.filter((a) => a !== false)

	//make new clause
	return clause(neg, pos)
}

function clauseTerm(a) {
	var neg = []
	var pos = []

	function rec(a) {
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

var a = logic.fn('a')
var b = logic.fn('b')
assert(logic.eq(clause([a], [b]), clause([a], [b])))
assert(logic.eq(clause([a], [b]), clauseTerm(logic.term('||', logic.term('!', a), b))))
assert(logic.eq(simplifyClause(clause([a], [b])), clause([a], [b])))

exports.clause = clause
