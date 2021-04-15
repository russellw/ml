'use strict'
const fs = require('fs')
const dimacs = require('./dimacs')
const tptp = require('./tptp')
const dpll = require('./dpll')

var lang
var files = []

function language(file) {
	if (lang) return lang
	switch (extension(file)) {
		case 'cnf':
			return 'dimacs'
		case 'p':
		case 'ax':
			return 'tptp'
	}
}

function extension(file) {
	var a = file.split('.')
	if (a.length < 2) return ''
	return a.pop()
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

function parseArgs(args) {
	for (var arg of args) {
		var s = arg
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
		if (extension(s) === 'lst') {
			parseArgs(fs.readFileSync(s, 'utf8').split(/\r?\n/))
			continue
		}
		files.push(s)
	}
}

parseArgs(process.argv.slice(2))
if (!files.length) {
	help()
	process.exit(0)
}
for (var file of files) {
	console.log(file)
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
	continue
	var r = dpll.sat(problem.clauses)
	if (r) {
		console.log('sat')
		var more
		for (var [k, v] of r.entries()) {
			if (more) process.stdout.write(' ')
			more = true
			if (!v) process.stdout.write('-')
			process.stdout.write(k)
		}
	} else console.log('unsat')
	console.log()
}
