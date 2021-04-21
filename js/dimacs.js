'use strict'
const etc = require('./etc')

var eof = ''

function parse(file, text) {
	var ti = 0
	var tok
	var toki
	var expected

	function err(msg) {
		etc.err(file, text, toki, msg)
	}

	// tokenizer
	function lex() {
		while (ti < text.length) {
			// mark start of token for error reporting
			toki = ti

			// space
			if (/\s/.test(text[ti])) {
				ti++
				continue
			}

			// line comment
			if (text[ti] === 'c') {
				while (text[ti] !== '\n') ti++
				if (!doneheader) console.log(text.slice(toki, ti))
				continue
			}

			// number
			if (/\d/.test(text[ti])) {
				while (/\d/.test(text[ti])) ti++
				tok = text.slice(toki, ti)
				return
			}

			// other
			tok = text[ti++]
			return
		}
		tok = eof
	}

	var doneheader
	lex()
	doneheader = true

	function eat(k) {
		if (tok === k) {
			lex()
			return true
		}
	}

	// parser
	var atoms = new Map()

	function atom() {
		if (!/[1-9]\d*/.test(tok)) err('Expected atom')
		var name = tok
		lex()
		// a propositional variable is a first-order function
		// for consistency, we use the first-order terminology throughout
		return etc.getor(atoms, name, () => {
			return { name }
		})
	}

	if (tok === 'p') {
		while (ti < text.length && /\s/.test(text[ti])) ti++
		if (text.slice(ti, ti + 3) !== 'cnf') {
			toki = ti
			err("Expected 'cnf'")
		}
		ti += 3
		lex()

		if (!/^\d+$/.test(tok)) err('Expected count')
		lex()

		if (!/^\d+$/.test(tok)) err('Expected count')
		lex()
	}

	var neg = []
	var pos = []
	var clauses = []

	function clause() {
		var c = [neg, pos]
		c.file = file
		clauses.push(c)
		neg = []
		pos = []
	}

	while (tok !== eof) {
		switch (tok) {
			case '0':
				lex()
				clause()
				continue
			case '-':
				lex()
				neg.push(atom())
				continue
		}
		pos.push(atom())
	}
	if (neg.length || pos.length) clause()
	return {
		formulas: [],
		clauses,
		expected,
	}
}

function prnsolution(m) {
	var more
	for (var [k, v] of m) {
		if (!k.name) continue
		if (more) process.stdout.write(' ')
		more = true
		if (!v) process.stdout.write('-')
		process.stdout.write(k.name)
	}
	console.log()
}

exports.parse = parse
exports.prnsolution = prnsolution
