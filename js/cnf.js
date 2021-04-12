'use strict'
var logic = require('./logic')

function clause(neg,pos){
	neg=neg.slice()
	neg.op='bag'

	pos=pos.slice()
	pos.op='bag'

	var c = [neg,pos]
	c.op = 'clause'
	return c
}

exports.clause = clause
exports.clause1 = clause1
