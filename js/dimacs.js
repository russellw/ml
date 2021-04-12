'use strict'
var logic = require('./logic')
var etc = require('./etc')

var eof = ' '

function parse(file, text) {
	var ti = 0
	var status = ''
	var tok
	var tokstart

	function err(msg) {
		etc.err(file, text, tokstart, msg)
	}

	// Tokenizer
	function lex() {
		while (ti < text.length) {
			//Mark start of token for error reporting
			tokstart = ti

			//space
			if (/\s/.test(text[ti])) {
				ti++
				continue
			}

			//line comment
			if (text[ti] == 'c') {
				while (text[ti] != '\n') ti++
				continue
			}

			//number
			if (/\d/.test(text[ti])) {
				while (/\d/.test(text[ti])) ti++
				tok = text.slice(tokstart, ti)
				return
			}

			//other
			tok = text[ti++]
			return
		}
		tok = eof
	}

	lex()

	function eat(k) {
		if (tok === k) {
			lex()
			return true
		}
	}

	// Parser
	var atoms = new Map()

	function atom() {
		if (!/[1-9]\d*/.test(tok)) err('Expected atom')
		var name = tok
		lex()
		if (atoms.has(name)) return atoms.get(name)
		var a = logic.fn(name)
		atoms.set(name, a)
		return a
	}

	if (eat('p')) {
		while (ti < text.length && /\s/.test(s[ti])) ti++
		if (text.slice(ti, ti + 3) !== 'cnf') {
			tokstart = ti
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
		clauses: cs,
		status,
	}
}

exports.parse = parse
