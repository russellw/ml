'use strict'
const assert = require('assert')
const etc = require('./etc')

class graph {
	constructor(nodes, arcs) {
		this.nodes = nodes
		this.arcs = new Map()
		for (var a of arcs) this.add(a[0], a[1])
	}

	add(x, y) {
		var arcs = this.arcs
		if (!arcs.has(x)) arcs.set(x, new Set())
		arcs.get(x).add(y)
	}

	dfswithout(x, f, w, visited) {
		if (x === w) return
		if (!visited) var visited = new Set()
		if (visited.has(x)) return
		visited.add(x)
		f(x)
		for (var y of this.successors(x)) this.dfswithout(y, f, w, visited)
	}

	domfrontier(s, x) {
		var u = []
		for (var y of this.nodes) {
			if (this.strictlydominates(s, x, y)) continue
			for (var z of this.predecessors(y))
				if (this.dominates(s, x, z)) {
					u.push(y)
					break
				}
		}
		return u
	}

	domtree(s) {
		var arcs = []
		for (var y of this.nodes) {
			var x = this.idom(s, y)
			if (x !== null) arcs.push([x, y])
		}
		return new graph(this.nodes, arcs)
	}

	dominates(s, x, y) {
		return !this.reacheswithout(s, y, x)
	}

	dominators(s, y) {
		return this.nodes.filter((x) => this.dominates(s, x, y))
	}

	idom(s, x) {
		for (var y of this.nodes) if (this.isidom(s, y, x)) return y
	}

	isidom(s, x, y) {
		if (!this.strictlydominates(s, x, y)) return
		for (var z of this.strictdominators(s, y)) if (!this.dominates(s, z, x)) return
		return 1
	}

	predecessors(y) {
		var u = []
		for (var a of this.arcs) {
			var [x, ys] = a
			if (ys.has(y)) u.push(x)
		}
		return u
	}

	reacheswithout(x, y, w) {
		if (w === x) return
		var r = null

		function f(z) {
			if (y === z) r = 1
		}

		this.dfswithout(x, f, w)
		return r
	}

	strictdominators(s, y) {
		return this.nodes.filter((x) => this.strictlydominates(s, x, y))
	}

	strictlydominates(s, x, y) {
		return x !== y && this.dominates(s, x, y)
	}

	successors(x) {
		if (!this.arcs.has(x)) return []
		return this.arcs.get(x)
	}
}

function test() {
	// https://tanujkhattar.wordpress.com/2016/01/11/dominator-tree-of-a-directed-graph/
	var nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'r']
	var arcs = [
		['a', 'd'],
		['b', 'a'],
		['b', 'd'],
		['b', 'e'],
		['c', 'f'],
		['c', 'g'],
		['d', 'l'],
		['e', 'h'],
		['f', 'i'],
		['g', 'i'],
		['g', 'j'],
		['h', 'e'],
		['h', 'k'],
		['i', 'k'],
		['j', 'i'],
		['k', 'i'],
		['k', 'r'],
		['l', 'h'],
		['r', 'a'],
		['r', 'b'],
		['r', 'c'],
	]
	var g = new graph(nodes, arcs)
	var s = 'r'

	// x dominates x
	for (var x of nodes) assert(g.dominates(s, x, x))

	// s dominates x
	for (var x of nodes) assert(g.dominates(s, s, x))

	// Immediate dominators
	for (var y of nodes) {
		var x = g.idom(s, y)
		switch (y) {
			case 'f':
			case 'g':
				assert(x === 'c')
				break
			case 'j':
				assert(x === 'g')
				break
			case 'l':
				assert(x === 'd')
				break
			case 'r':
				assert(!x)
				break
			default:
				assert(x === 'r')
				break
		}
	}

	// Tiger book page 439
	var nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
	var arcs = [
		[1, 2],
		[1, 5],
		[1, 9],
		[2, 3],
		[3, 3],
		[3, 4],
		[4, 13],
		[5, 6],
		[5, 7],
		[6, 4],
		[6, 8],
		[7, 8],
		[7, 12],
		[8, 13],
		[8, 5],
		[9, 10],
		[9, 11],
		[10, 12],
		[11, 12],
		[12, 13],
	]
	var g = new graph(nodes, arcs)
	var s = 1

	// Dominance frontier
	assert(etc.eqsets(new Set(g.domfrontier(s, 5)), new Set([4, 5, 12, 13])))
}

test()

module.exports = graph
