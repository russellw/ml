'use strict'
var logic = require('./logic')
var assert = require('assert')

var falseClause = [[], []]
var trueClause = [[], [true]]

//var falseClauses =

function clause(neg, pos, m = new Map()) {
	//simplify
	neg = neg.map((a) => logic.simplify(a, m))
	pos = pos.map((a) => logic.simplify(a, m))

	//filter out redundancy
	neg = neg.filter((a) => a !== true)
	pos = pos.filter((a) => a !== false)

	//tautology?
	for (var a in neg) if (a === false) return trueClause
	for (var a in pos) if (a === true) return trueClause
	for (var a in neg) for (var b in pos) if (logic.eq(a, b)) return trueClause

	//make new clause
	var c = [neg, pos]
	c.op = 'clause'
	return c
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

var a = { op: 'fn' }
var b = { op: 'fn' }
assert(logic.eq(clause([a], [b]), clause([a], [b])))
assert(logic.eq(clause([a], [b]), clauseTerm(logic.term('||', logic.term('!', a), b))))

exports.clause = clause
