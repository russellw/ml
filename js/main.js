'use strict'
var fs = require('fs')
var dimacs = require('./dimacs')

var lang

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
	console.log('Options:')
	console.log()
	console.log('-h  Show help')
	console.log('-v  Show version')
}

function version() {
	console.log('Version 0')
}

var files = []
for (var arg of process.argv.slice(2)) {
	var s = arg
	if (!s.startsWith('-')) {
		files.push(s)
		continue
	}
	while (s.startsWith('-')) s = s.slice(1)
	switch (s) {
		case 'h':
		case 'help':
			help()
			continue
		case 'h':
		case 'version':
			version()
			continue
	}
	console.error(arg + ': unknown option')
	process.exit(1)
}
for (var file of files) {
	var text = fs.readFileSync(file, 'utf8')
	dimacs.parse(file, text)
}
