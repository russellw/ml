'use strict'
const fs = require('fs')
const dimacs = require('./dimacs')
const tptp = require('./tptp')
const dpll = require('./dpll')
const etc = require('./etc')
const cnf = require('./cnf')
const superposition = require('./superposition')
const assert = require('assert')

var lang
var files = []

function language(file) {
	if (lang) return lang
	switch (etc.extension(file)) {
		case 'cnf':
			return 'dimacs'
		case 'p':
		case 'ax':
			return 'tptp'
	}
}

function help() {
	console.log('General options:')
	console.log('-h       Show help')
	console.log('-v       Show version')
	console.log()
	console.log('Input:')
	console.log('-dimacs  DIMACS format')
	console.log('-tptp    TPTP   format')
}

function version() {
	console.log('Version 0')
}

function parseargs(args) {
	for (var arg of args) {
		var s = arg
		if (!s) continue
		if (s.startsWith('-')) {
			while (s.startsWith('-')) s = s.slice(1)
			switch (s) {
				case 'dimacs':
				case 'tptp':
					lang = s
					continue
				case 'h':
				case 'help':
					help()
					process.exit(0)
				case 'h':
				case 'version':
					version()
					process.exit(0)
			}
			console.error(arg + ': unknown option')
			process.exit(1)
		}
		if (etc.extension(s) === 'lst') {
			parseargs(fs.readFileSync(s, 'utf8').split(/\r?\n/))
			continue
		}
		files.push(s)
	}
}

function propositional(clauses) {
	for (var c of clauses) for (var L of c) for (var a of L) if (Array.isArray(a)) return
	return true
}

function solve(problem) {
	for (var c of problem.formulas) cnf.convert(c, problem.clauses)
	if (propositional(problem.clauses)) return dpll.solve(problem.clauses)
	return superposition.solve(problem.clauses)
}

function test() {
	function sat(cs) {
		var r1 = superposition.solve(cs).sat
		if (propositional(cs)) {
			var r2 = dpll.solve(cs).sat
			assert(r1 === r2)
		}
		return r1
	}

	function thm(a) {
		var cs = []
		cnf.convert([a], cs)
		assert(sat(cs))

		var cs = []
		cnf.convert([etc.mk('!', a)], cs)
		assert(!sat(cs))
	}

	var a = { type: 'boolean' }
	var b = { type: 'boolean' }

	thm(true)
	thm(etc.mk('=>', false, a))
	thm(etc.mk('=>', a, a))
	thm(etc.mk('&&', true, true, true))
	thm(etc.mk('||', false, false, true))
	thm(etc.mk('<=>', a, a))

	var p1 = { name: 'p1', type: 'boolean' }
	var p2 = { name: 'p2', type: 'boolean' }
	var p3 = { name: 'p3', type: 'boolean' }

	thm(etc.mk('<=>', p1, etc.mk('<=>', p2, etc.mk('<=>', p1, p2))))

	function eqv(a, b) {
		thm(etc.mk('<=>', a, b))
	}

	eqv(etc.mk('!', etc.mk('!', a)), a)
	eqv(etc.mk('&&', a, b), etc.mk('&&', b, a))
	eqv(etc.mk('||', a, b), etc.mk('||', b, a))
	eqv(etc.mk('<=>', a, b), etc.mk('<=>', b, a))
	eqv(etc.mk('!', etc.mk('<=>', a, b)), etc.mk('!', etc.mk('<=>', b, a)))

	thm(etc.mk('=>', etc.mk('&&', a, etc.mk('=>', a, b)), b))

	var a = { type: 'individual' }
	var b = { type: 'individual' }
	var f1 = { type: ['individual', 'individual'] }
	var f2 = { type: ['individual', 'individual', 'individual'] }
	var g1 = { type: ['individual', 'individual'] }
	var g2 = { type: ['individual', 'individual', 'individual'] }
	var x = { o: 'var', type: 'individual' }
	var y = { o: 'var', type: 'individual' }
	var z = { o: 'var', type: 'individual' }

	function imp(a, b) {
		thm(etc.mk('=>', a, b))
	}

	function eq(a, b) {
		return etc.mk('==', a, b)
	}

	var fa = etc.mk('call', f1, a)
	var fb = etc.mk('call', f1, b)
	var fx = etc.mk('call', f1, x)
	var fy = etc.mk('call', f1, y)

	imp(eq(a, b), eq(fa, fb))
	thm(etc.mk('all', [x, y], etc.mk('=>', eq(x, y), eq(fx, fy))))
}

test()

if (require.main === module) {
	parseargs(process.argv.slice(2))
	if (!files.length) {
		help()
		process.exit(0)
	}
	for (var file of files) {
		try {
			var text = fs.readFileSync(file, 'utf8')
			switch (language(file)) {
				case 'dimacs':
					var problem = dimacs.parse(file, text)
					break
				case 'tptp':
					var problem = tptp.parse(file, text)
					break
				default:
					console.error(file + ': unknown language')
					process.exit(1)
			}
		} catch (e) {
			if (e === 'Inappropriate') {
				console.log('%% SZS status Inappropriate for ' + file)
				console.log()
				continue
			}
			if (e.code === 'ERR_STRING_TOO_LONG' || e.message === 'Array buffer allocation failed') {
				console.log('%% SZS status ResourceOut for ' + file)
				console.log()
				continue
			}
			throw e
		}
		var r = solve(problem)
		switch (language(file)) {
			case 'dimacs':
				switch (r.sat) {
					case true:
						console.log('sat')
						if (r.solution) {
							var more
							for (var [k, v] of r.solution) {
								if (!k.name) continue
								if (more) process.stdout.write(' ')
								more = true
								if (!v) process.stdout.write('-')
								process.stdout.write(k.name)
							}
							console.log()
						}
						break
					case false:
						console.log('unsat')
						break
				}
				break
		}
		console.log()
	}
}
