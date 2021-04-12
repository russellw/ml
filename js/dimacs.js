'use strict'
const logic = require('./logic')
const etc = require('./etc')

const eof = ' '

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
			if (etc.isspace(text[ti])) {
				ti++
				continue
			}

			//comment
			if (text[ti] == 'c') {
				while (text[ti] != '\n') ti++
				continue
			}

			//number
			if (etc.isdigit(text[ti])) {
				while (etc.isdigit(text[ti])) ti++
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
	const atoms = new Map()

	function atom() {
		if (!('1' <= tok[0] && tok[0] <= '9')) err('Expected atom')
		const name = tok
		lex()
		if (atoms.has(name)) return atoms.get(name)
		const a = logic.fn(name)
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
	const clauses = []

	function clause() {
		const c = [neg, pos]
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
