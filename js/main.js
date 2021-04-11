'use strict'
var logic = require('./logic')

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
	console.log('%s: unknown option', arg)
	process.exit(1)
}
