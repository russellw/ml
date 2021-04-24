'use strict'
const fs = require('fs')
const etc = require('./etc')

var files = []

function walk(file) {
	if (fs.statSync(file).isDirectory()) {
		for (var fi of fs.readdirSync(file)) walk(file + '/' + fi)
		return
	}
	switch (etc.extension(file)) {
		case 'p':
		case 'ax':
			files.push(file)
			break
	}
}

walk('/TPTP')
// walk('a.p')
var total = new Map()

function add(k, n) {
	if (!total.has(k)) total.set(k, 0)
	n = parseInt(n)
	total.set(k, total.get(k) + n)
}

function esc(k) {
	var r = []
	for (var c of k) r.push('\\' + c)
	return r.join('')
}

for (var file of files) {
	try {
		var lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
		var ks = ['~', '&', '~&', '|', '~|', '=>', '<=', '<=>', '<~>']
		for (var i = 0; i < lines.length; i++) {
			if (/Number of connectives/.test(lines[i])) {
				for (var s of lines.slice(i, i + 7)) {
					for (var k of ks) {
						var re = new RegExp('(\\d+)\\s+' + esc(k) + '[;\\)]')
						var m = re.exec(s)
						if (m) {
							add(k, m[1])
						}
					}
				}
			}
		}
	} catch (e) {
		var r
		if (typeof e === 'string') r = { szs: e }
		else if (e.code === 'ERR_STRING_TOO_LONG' || e.message === 'Array buffer allocation failed') r = { szs: 'ResourceOut' }
		else throw e
	}
}
console.log(total)
