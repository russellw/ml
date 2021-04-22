'use strict'
const path = require('path')
const fs = require('fs')
const dimacs = require('./dimacs')
const tptp = require('./tptp')
const dpll = require('./dpll')
const etc = require('./etc')
const cnf = require('./cnf')
const superposition = require('./superposition')
const assert = require('assert')

var lang
var timelimit
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
	console.log('-h          show help')
	console.log('-V          show version')
	console.log()
	console.log('Input:')
	console.log('-dimacs     DIMACS format')
	console.log('-tptp       TPTP   format')
	console.log()
	console.log('Resources:')
	console.log('-t seconds  time limit')
}

function parseargs(args) {
	for (var i = 0; i < args.length; i++) {
		var s = args[i]
		if (!s) continue
		if (s.startsWith('-')) {
			while (s.startsWith('-')) s = s.slice(1)

			var optarg
			var m = /^([a-zA-Z\-])+[:=](.+)$/.exec(s)
			if (m) {
				s = m[1]
				optarg = m[2]
			} else {
				var m = /^([a-zA-Z\-])+(\d+)$/.exec(s)
				if (m) {
					s = m[1]
					optarg = m[2]
				}
			}

			function getoptarg() {
				if (optarg) return optarg
				if (i + 1 === args.length) {
					console.error(args[i] + ': expected arg')
					process.exit(1)
				}
				return args[++i]
			}

			switch (s) {
				case 't':
				case 'soft-cpu-limit':
					timelimit = parseFloat(getoptarg()) * 1000
					continue
				case 'dimacs':
				case 'tptp':
					lang = s
					continue
				case 'h':
				case 'help':
					help()
					process.exit(0)
				case 'V':
				case 'v':
				case 'version':
					console.log('Aklo version %s', etc.version)
					process.exit(0)
			}
			console.error(args[i] + ': unknown option')
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

function solve(problem, deadline) {
	for (var c of problem.formulas) cnf.convert(c, problem.clauses)
	if (propositional(problem.clauses)) return dpll.solve(problem.clauses, deadline)
	return superposition.solve(problem.clauses, deadline)
}

function test() {
	function sat(cs) {
		var r1 = superposition.solve(cs).szs
		if (propositional(cs)) {
			var r2 = dpll.solve(cs).szs
			assert(r1 === r2)
		}
		return r1
	}

	function thm(a) {
		var cs = []
		cnf.convert([a], cs)
		assert(sat(cs) === 'Satisfiable')

		var cs = []
		cnf.convert([etc.mk('!', a)], cs)
		assert(sat(cs) === 'Unsatisfiable')
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
		var start = new Date().getTime()
		var deadline
		if (timelimit) deadline = start + timelimit
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
			var r = solve(problem, deadline)
		} catch (e) {
			if (typeof e === 'string') r = { szs: e }
			else if (e.code === 'ERR_STRING_TOO_LONG' || e.message === 'Array buffer allocation failed') r = { szs: 'ResourceOut' }
			else throw e
		}
		if (problem.conjecture)
			switch (r.szs) {
				case 'Satisfiable':
					r.szs = 'CounterSatisfiable'
					break
				case 'Unsatisfiable':
					r.szs = 'Theorem'
					break
			}
		console.log('%% SZS status %s for %s', r.szs, path.basename(file))
		if (r.proof) {
			console.log('%% SZS output start CNFRefutation for %s', path.basename(file))
			tptp.prnproof(r.proof)
			console.log('%% SZS output end CNFRefutation for %s', path.basename(file))
		}
		if (problem.expected && r.szs !== problem.expected)
			switch (r.szs) {
				case 'Unsatisfiable':
				case 'Theorem':
					if (problem.expected === 'ContradictoryAxioms') break
				case 'Satisfiable':
				case 'CounterSatisfiable':
					console.error(r.szs + ' != ' + problem.expected)
					process.exit(1)
			}
		console.log('%% %d seconds', (new Date().getTime() - start) / 1000)
		console.log()
	}
}

exports.solve = solve
