//reduce entropy of javascript code
//does not work on arbitrary javascript!
//should be inspected carefully before being used in other projects
'use strict'
const fs = require('fs')

function extension(file) {
	var a = file.split('.')
	if (a.length < 2) return ''
	return a.pop()
}

function eq(a, b) {
	if (a.length != b.length) return
	for (var i = 0; i < a.length; i++) if (a[i] != b[i]) return
	return true
}

if (process.argv[2] !== '.') process.exit(1)
for (var file of fs.readdirSync('.')) {
	if (extension(file) !== 'js') continue
	var lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
	var old = lines.slice()
	for (var i = 0; i < lines.length; i++) {
		var m

		m = /var (\w+ = require\('.+'\))/.exec(lines[i])
		if (m) {
			lines[i] = 'const ' + m[1]
			console.log(lines[i])
		}
	}
	if (eq(lines, old)) continue
	fs.writeFileSync(file, lines.join('\n'), 'utf8')
	console.log(file)
}
