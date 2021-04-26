'use strict'
const assert = require('assert')

class priorityq {
	constructor(priority = (a) => a) {
		this.priority = priority
		this.data = []
	}

	push(a) {
		this.data.push(a)
	}

	pop() {
		if (!this.data.length) return
		var i = 0
		var a = this.data[i]
		for (var j = 1; j < this.data.length; j++)
			if (this.priority(this.data[j]) < this.priority(a)) {
				i = j
				a = this.data[j]
			}
		this.data.splice(i, 1)
		return a
	}
}

function test() {
	var q = new priorityq()
	q.push(5)
	q.push(1)
	q.push(4)
	q.push(2)
	q.push(3)
	assert(q.pop() === 1)
	assert(q.pop() === 2)
	assert(q.pop() === 3)
	assert(q.pop() === 4)
	assert(q.pop() === 5)
	assert(!q.pop())
}

test()

module.exports = priorityq
