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

	function eat(k) {
		if (tok === k) {
			lex()
			return true
		}
	}

	// Parser
	var atoms = new Map()

	function atom() {
		if (!('1' <= tok[0] && tok[0] <= '9')) err('Expected atom')
		var name = tok
		lex()
		if (atoms.has(name)) return atoms.get(name)
		var a = cnf.fun(name)
		atoms.set(name, a)
		return a
	}
}
