'use strict'
const path = require('path')
const etc = require('./etc')

var eof = ''

function parse(file, text) {
	var ti = 0
	var tok
	var expected

	function err(msg) {
		console.error('%s:%d: %s', file, text.slice(0, ti).split('\n').length, msg)
		process.exit(1)
	}

	// tokenizer
	function lex() {
		while (ti < text.length) {
			// mark start of token for error reporting
			var ti0 = ti

			// space
			if (/\s/.test(text[ti])) {
				ti++
				continue
			}

			// line comment
			if (text[ti] === 'c') {
				while (text[ti] !== '\n') ti++
				var s = text.slice(ti0, ti)
				if (!doneheader) console.log('%' + s.slice(1))
				if (!expected) {
					var m = /^c.* (SAT|UNSAT) /.exec(s)
					if (m) expected = m[1] === 'SAT' ? 'Satisfiable' : 'Unsatisfiable'
				}
				continue
			}

			// number
			if (/\d/.test(text[ti])) {
				while (/\d/.test(text[ti])) ti++
				tok = text.slice(ti0, ti)
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
		if (!/[1-9]\d*/.test(tok)) err('expected atom')
		var name = tok
		lex()
		// a propositional variable is a first-order function
		// for consistency, we use the first-order terminology throughout
		return etc.getor(atoms, name, () => {
			return { o: 'fn', name, type: 'boolean' }
		})
	}

	if (tok === 'p') {
		while (ti < text.length && /\s/.test(text[ti])) ti++
		if (text.slice(ti, ti + 3) !== 'cnf') err("expected 'cnf'")
		ti += 3
		lex()

		if (!/^\d+$/.test(tok)) err('expected count')
		lex()

		if (!/^\d+$/.test(tok)) err('expected count')
		lex()
	}

	var neg = []
	var pos = []
	var clauses = []

	function clause() {
		var c = [neg, pos]
		c.file = path.basename(file)
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

exports.parse = parse
