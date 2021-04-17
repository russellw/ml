// reduce entropy of javascript code
// assumes prettier has already been run
// does not work on arbitrary javascript!
// should be inspected carefully before being used in other projects
'use strict'
const fs = require('fs')
const os = require('os')
const etc = require('./etc')

function quote(s) {
	var q = ''
	for (var i = 0; i < s.length; i++) {
		if (s.slice(i, i + 2) === '//' && !q) return '//'
		switch (s[i]) {
			case '\\':
				i++
				break
			case '"':
			case "'":
			case '/':
				if (!q) {
					q = s[i]
					break
				}
				if (q === s[i]) q = ''
				break
		}
	}
	return q
}

if (process.argv[2] !== '.') process.exit(1)
for (var file of fs.readdirSync('.')) {
	if (etc.extension(file) !== 'js') continue
	var lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
	var old = lines.slice()
	for (var i = 0; i < lines.length; i++) {
		// comments begin with spaces
		var m = /^(\s*)\/\/(\S.*)$/.exec(lines[i])
		if (m) lines[i] = m[1] + '// ' + m[2]

		// var ... require -> const
		var m = /^var (\w+ = require\('.+'\))$/.exec(lines[i])
		if (m) lines[i] = 'const ' + m[1]

		// for ... in -> of
		var m = /^(.*)for (.*) in (.*)$/.exec(lines[i])
		if (m && !quote(m[1]) && !quote(m[2])) lines[i] = m[1] + 'for ' + m[2] + ' of ' + m[3]

		// == -> ===
		var m = /^(.*) == (.*)$/.exec(lines[i])
		if (m && !quote(m[1])) lines[i] = m[1] + ' === ' + m[2]

		// != -> !==
		var m = /^(.*) != (.*)$/.exec(lines[i])
		if (m && !quote(m[1])) lines[i] = m[1] + ' !== ' + m[2]
	}
	if (etc.eq(lines, old)) continue
	fs.renameSync(file, os.tmpdir() + '/' + file)
	fs.writeFileSync(file, lines.join('\n'), 'utf8')
	console.log(file)
}
