'use strict'
const path = require('path')
const fs = require('fs')
const dimacs = require('./aklo/dimacs')
const tptp = require('./aklo/tptp')
const solver = require('./aklo/solver')
const etc = require('./aklo/etc')
const assert = require('assert')

var lang = null
var timelimit = null
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
	console.log('-           read stdin')
	console.log()
	console.log('Resources:')
	console.log('-t seconds  time limit')
}

function parseargs(args) {
	for (var i = 0; i < args.length; i++) {
		var s = args[i]
		if (!s) continue
		if (s === '-') {
			files.push('stdin')
			continue
		}
		if (s.startsWith('-')) {
			while (s.startsWith('-')) s = s.slice(1)

			var optarg = null
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

	parseargs(process.argv.slice(2))
	if (!files.length) {
		help()
		process.exit(0)
	}
	var start = new Date().getTime()
	var solved = 0
	for (var file of files) {
		var start1 = new Date().getTime()
		var deadline = null
		if (timelimit) deadline = start1 + timelimit
		try {
			var txt = fs.readFileSync(file === 'stdin' ? 0 : file, 'utf8')
			switch (language(file)) {
				case 'dimacs':
					var problem = dimacs.parse(file, txt)
					break
				case 'tptp':
					var problem = tptp.parse(file, txt)
					break
				default:
					console.error(file + ': unknown language')
					process.exit(1)
			}
			var r = solver.solve(problem, deadline)
		} catch (e) {
			if (typeof e === 'string') r = { szs: e }
			else if (e.code === 'ERR_STRING_TOO_LONG') r = { szs: 'ResourceOut' }
			else if (e.message === 'Array buffer allocation failed') r = { szs: 'MemoryOut' }
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
		switch (r.szs) {
			case 'Unsatisfiable':
			case 'Theorem':
			case 'Satisfiable':
			case 'CounterSatisfiable':
				solved++
				break
		}
		console.log('%% %d seconds', (new Date().getTime() - start1) / 1000)
		console.log()
	}
	console.log('%% Solved %d/%d(%d%%)', solved, files.length, (solved / files.length) * 100)
	console.log('%% %d seconds', (new Date().getTime() - start) / 1000)
