// reduce entropy of javascript code
// assumes prettier has already been run
// does not work on arbitrary javascript!
// should be inspected carefully before being used in other projects
'use strict'
const fs = require('fs')
const os = require('os')
const path = require('path')

// copy paste some standard library code instead of 'require'ing it
// to preserve the ability to run this program when in the middle of editing the standard library

function eq(a, b) {
	if (a === b) return true
	if (!Array.isArray(a)) return
	if (a.o !== b.o) return
	if (a.length !== b.length) return
	for (var i = 0; i < a.length; i++) if (!eq(a[i], b[i])) return
	return true
}

function walkfiles(files, filter, act) {
	function rec(dir) {
		var fis = fs.readdirSync(dir)
		for (var fi of fis) {
			var file = dir + '/' + fi
			if (fs.statSync(file).isDirectory()) rec(file)
			else if (filter(file)) act(file)
		}
	}

	for (var file of files)
		if (fs.statSync(file).isDirectory()) rec(file)
		else act(file)
}

function extension(file) {
	var a = file.split('.')
	if (a.length < 2) return ''
	return a.pop()
}

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

function dofile(file) {
	var lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
	var old = [...lines]

	// use strict
	for (var i = 0; i < lines.length; i++) if (lines[i] && !lines[i].startsWith('//')) break
	if (lines[i] !== "'use strict'") lines.splice(i, 0, "'use strict'")

	// individual lines
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

		// var x -> var x = null
		var m = /^\s*var \w+$/.exec(lines[i])
		if (m) lines[i] += ' = null'
	}

	// module.exports
	for (var i = lines.length; i && !lines[i - 1]; ) i--
	var ex = []
	for (; i; i--) {
		var m = /^exports\.(\w*) = (\w*)$/.exec(lines[i - 1])
		if (!m) break
		if (m[1] !== m[2]) throw lines[i - 1]
		ex.push(m[1])
	}
	if (ex.length) {
		lines.splice(i, lines.length - i)
		lines.push('module.exports = {')
		for (var s of ex) lines.push('\t' + s + ',')
		lines.push('}')
		lines.push('')
	}

	// save
	if (eq(lines, old)) return
	fs.renameSync(file, os.tmpdir() + '/' + path.basename(file))
	fs.writeFileSync(file, lines.join('\n'), 'utf8')
	console.log(file)
}

walkfiles(process.argv.slice(2), (file) => extension(file) == '.js', dofile)
