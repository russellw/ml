'use strict'
const assert = require('assert')

function mk(priority = (a) => a) {
	return {
		priority,
		data: [],
	}
}

function push(q, a) {
	q.data.push(a)
}

function pop(q) {
	if (!q.data.length) return null
	var i = 0
	var a = q.data[i]
	for (var j = 1; j < q.data.length; j++)
		if (q.priority(q.data[j]) < q.priority(a)) {
			i = j
			a = q.data[j]
		}
	q.data.splice(i, 1)
	return a
}

function test() {
	var q = mk()
	push(q, 5)
	push(q, 1)
	push(q, 4)
	push(q, 2)
	push(q, 3)
	assert(pop(q) === 1)
	assert(pop(q) === 2)
	assert(pop(q) === 3)
	assert(pop(q) === 4)
	assert(pop(q) === 5)
	assert(pop(q) === null)
}

test()

exports.mk = mk
exports.push = push
exports.pop = pop
