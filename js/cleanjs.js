'use strict'
var fs = require('fs')

function extension(file) {
	var a = file.split('.')
	if (a.length < 2) return ''
	return a.pop()
}

if (process.argv[2] !== '.') process.exit(1)
for (var file of fs.readdirSync('.')) {
	if (extension(file) !== 'js') continue
	var lines = fs.readFileSync(s, 'utf8').split(/\r?\n/)
}
