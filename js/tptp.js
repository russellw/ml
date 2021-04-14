'use strict'
const logic = require('./logic')
const etc = require('./etc')
const cnf = require('./cnf')
const fs = require('fs')

var eof = ''

function unquote(s) {
	s = s.slice(1, s.length - 1)
	var r = []
	for (var i = 0; i < s.length; i++) {
		if (s[i] === '\\') i++
		r.push(s[i])
	}
	return r.join('')
}

function parse1(file, text, selection, problem) {
	var ti = 0
	var tok
	var toki

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
			if (text[ti] === '%') {
				while (text[ti] !== '\n') ti++
				console.log(text.slice(toki, ti))
				continue
			}

			// block comment
			if (text.slice(ti, ti + 2) === '/*') {
				for (ti += 2; text.slice(ti, ti + 2) !== '*/'; ti++) if (ti === text.length) err("Unclosed '/*'")
				continue
			}

			// word
			if (/[\w_\$]/.test(text[ti])) {
				while (/[\w_\$]/.test(text[ti])) ti++
				tok = text.slice(toki, ti)
				return
			}

			// quote
			if (text[ti] === "'" || text[ti] === '"') {
				for (var q = text[ti++]; text[ti] !== q; ti++) {
					if (text[ti] === '\\') ti++
					if (ti === text.length) err('Unclosed quote')
				}
				ti++
				tok = text.slice(toki, ti)
				return
			}

			// number
			if (/[\+\-]?\d/.test(text.slice(ti, ti + 2))) {
				// sign
				if (/[\+\-]/.test(text[ti])) ti++

				// integer
				while (/\d/.test(text[ti])) ti++

				// fraction
				if (text[ti] === '/') {
					// denominator
					ti++
					while (/\d/.test(text[ti])) ti++
				} else {
					// decimal
					if (text[ti] === '.') {
						ti++
						while (/\d/.test(text[ti])) ti++
					}

					// exponent
					if (/[\+\-]?[Ee]/.test(text.slice(ti, ti + 2))) {
						if (/[\+\-]/.test(text[ti])) ti++
						if (/[Ee]/.test(text[ti])) ti++
						while (/\d/.test(text[ti])) ti++
					}
				}
				tok = text.slice(toki, ti)
				return
			}

			// 3-char punctuation
			var punct = ['<=>', '<~>']
			if (punct.includes(text.slice(ti, ti + 3))) {
				ti += 3
				tok = text.slice(toki, ti)
				return
			}

			// 2-char punctuation
			var punct = ['!=', '=>', '<=', '~|', '~&']
			if (punct.includes(text.slice(ti, ti + 2))) {
				ti += 2
				tok = text.slice(toki, ti)
				return
			}

			// other
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

	function expect(k) {
		if (!eat(k)) err("Expected '" + k + "'")
	}

	// types
	function atomicType() {
		switch (tok) {
			case '!':
			case '[':
				throw 'Inappropriate'
			case '(':
				lex()
				var t = atomicType()
				expect(')')
				return t
			case '$i':
				lex()
				return 'individual'
			case '$o':
				lex()
				return 'bool'
			case '$int':
				lex()
				return 'integer'
			case '$rat':
				lex()
				return 'rational'
			case '$real':
				lex()
				return 'real'
		}
		if (/^[\w_]+/.test(tok)) {
			var t = tok
			lex()
			return t
		}
		if (tok[0] === "'") {
			var t = tok
			lex()
			return unquote(t)
		}
		err('Expected type')
	}

	// //////////////////

	// Parser
	var free

	function annotated_formula() {
		lex()

		// Name
		expect('(')
		if (select(formula_name())) {
			// Role
			expect(',')
			if (!tok) err('Expected role')
			if (!iop.islower(tok[0])) err('Expected role')
			var role = tok
			lex()

			// Formula
			expect(',')
			free = new Map()
			var a = formula(cnf.empty)
			if (free.size) a = cnf.quant('!', Array.from(free.values()), a)
			if (role === 'conjecture') {
				if (conjecture) err('Multiple conjectures not supported')
				a = cnf.term('~', a)
				conjecture = a
			}
			formulas.push(a)
		}

		// Annotations
		if (eat(',')) while (tok !== ')') ignore()

		// End
		expect(')')
		expect('.')
	}

	function defined_term(bound) {
		switch (tok) {
			case '$difference':
				return defined_term_arity(bound, '-', 2)
			case '$distinct':
				var args = term_args(bound)
				var clauses = cnf.term('&')
				for (var i = 0; i < args.length; i++) for (var j = 0; j < i; j++) clauses.push(cnf.term('!=', args[i], args[j]))
				return clauses
			case '$false':
				lex()
				return cnf.bool(false)
			case '$greater':
				return defined_term_arity(bound, '>', 2)
			case '$greatereq':
				return defined_term_arity(bound, '>=', 2)
			case '$less':
				return defined_term_arity(bound, '<', 2)
			case '$lesseq':
				return defined_term_arity(bound, '<=', 2)
			case '$product':
				return defined_term_arity(bound, '*', 2)
			case '$quotient':
				return defined_term_arity(bound, '/', 2)
			case '$sum':
				return defined_term_arity(bound, '+', 2)
			case '$true':
				lex()
				return cnf.bool(true)
			case '$uminus':
				return defined_term_arity(bound, '-', 1)
		}
		err('Unknown term')
	}

	function defined_term_arity(bound, op, arity) {
		var args = term_args(bound)
		if (args.length !== arity) err('Expected ' + arity + ' arguments')
		return cnf.term(op, ...args)
	}

	function formula(bound) {
		var args = [unitary_formula(bound)]
		var op = tok
		switch (tok) {
			case '&':
			case '|':
				while (eat(op)) args.push(unitary_formula(bound))
				break
			case '<=':
				lex()
				args.unshift(unitary_formula(bound))
				op = '=>'
				break
			case '<=>':
			case '<~>':
			case '=>':
			case '~&':
			case '~|':
				lex()
				args.push(unitary_formula(bound))
				break
			default:
				return args[0]
		}
		return cnf.term(op, ...args)
	}

	function formula_name() {
		if (!tok) err('Expected formula name')
		switch (tok[0]) {
			case "'":
				var name = unquote(tok)
				lex()
				return name
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
			case 'a':
			case 'b':
			case 'c':
			case 'd':
			case 'e':
			case 'f':
			case 'g':
			case 'h':
			case 'i':
			case 'j':
			case 'k':
			case 'l':
			case 'm':
			case 'n':
			case 'o':
			case 'p':
			case 'q':
			case 'r':
			case 's':
			case 't':
			case 'u':
			case 'v':
			case 'w':
			case 'x':
			case 'y':
			case 'z':
				var name = tok
				lex()
				return name
		}
		err('Expected name')
	}

	function ignore() {
		if (!tok) err("Expected ')'")
		switch (tok) {
			case '(':
				lex()
				while (!eat(')')) ignore()
				return
			case '[':
				lex()
				while (!eat(']')) {
					if (!tok) err("Expected ']'")
					ignore()
				}
				return
		}
		lex()
	}

	function include() {
		lex()

		// File
		expect('(')
		if (!tok.startsWith("'")) err('Expected file')
		var name = unquote(tok)
		lex()

		// Selection
		var selection1 = selection
		if (eat(',')) {
			expect('[')
			selection1 = new Set()
			do {
				var s = formula_name()
				if (select(s)) selection1.add(s)
			} while (eat(','))
			expect(']')
		}

		// End
		expect(')')
		expect('.')

		// Absolute
		if (path.isAbsolute(name)) {
			var file1 = name
			var text1 = fs.readFileSync(file1, 'utf8')
			parse1(text1, file1, selection1)
			return
		}

		// Relative
		var tptp = process.env.TPTP
		if (!tptp) err('TPTP environment variable not defined')
		var file1 = tptp + '/' + name
		var text1 = fs.readFileSync(file1, 'utf8')
		parse1(text1, file1, selection1)
	}

	function infix_unary(bound) {
		var a = term(bound)
		switch (tok) {
			case '!=':
			case '=':
				var op = tok
				lex()
				return cnf.term(op, a, term(bound))
		}
		return a
	}

	function plain_term(bound, name) {
		lex()
		var f = funs.get(name)
		if (!f) {
			f = cnf.fun(name)
			funs.set(name, f)
		}
		if (tok !== '(') return f
		var args = term_args(bound)
		return cnf.call(f, args)
	}

	function select(name) {
		if (!selection) return true
		return selection.has(name)
	}

	function term(bound) {
		if (!tok) err('Expected term')
		switch (tok[0]) {
			case '"':
				var name = unquote(tok)
				lex()
				a = distinct_objs.get(name)
				if (a) return a
				a = cnf.distinct_obj(name)
				distinct_objs.set(name, a)
				return a
			case '$':
				return defined_term(bound)
			case "'":
				return plain_term(bound, unquote(tok))
			case '+':
			case '-':
				if (!iop.isdigit(tok[1])) break
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
				var a = value
				lex()
				return a
			case 'A':
			case 'B':
			case 'C':
			case 'D':
			case 'E':
			case 'F':
			case 'G':
			case 'H':
			case 'I':
			case 'J':
			case 'K':
			case 'L':
			case 'M':
			case 'N':
			case 'O':
			case 'P':
			case 'Q':
			case 'R':
			case 'S':
			case 'T':
			case 'U':
			case 'V':
			case 'W':
			case 'X':
			case 'Y':
			case 'Z':
				var name = tok
				lex()
				var a = bound.get(name)
				if (a) return a
				a = free.get(name)
				if (a) return a
				a = cnf.variable(name)
				free.set(name, a)
				return a
			case 'a':
			case 'b':
			case 'c':
			case 'd':
			case 'e':
			case 'f':
			case 'g':
			case 'h':
			case 'i':
			case 'j':
			case 'k':
			case 'l':
			case 'm':
			case 'n':
			case 'o':
			case 'p':
			case 'q':
			case 'r':
			case 's':
			case 't':
			case 'u':
			case 'v':
			case 'w':
			case 'x':
			case 'y':
			case 'z':
				return plain_term(bound, tok)
		}
		err('Expected term')
	}

	function term_args(bound) {
		expect('(')
		var a = [term(bound)]
		while (eat(',')) a.push(term(bound))
		expect(')')
		return a
	}

	function unitary_formula(bound) {
		switch (tok) {
			case '!':
			case '?':
				var op = tok
				lex()
				expect('[')
				var variables = []
				do {
					var a = cnf.variable(tok)
					bound = bound.add(tok, a)
					lex()
				} while (eat(','))
				expect(']')
				expect(':')
				return cnf.quant(op, variables, unitary_formula(bound))
			case '(':
				lex()
				var a = formula(bound)
				expect(')')
				return a
			case '~':
				lex()
				return cnf.term('~', unitary_formula(bound))
		}
		return infix_unary(bound)
	}

	while (tok)
		switch (tok) {
			case 'cnf':
			case 'fof':
				annotated_formula()
				break
			case 'include':
				include()
				break
			default:
				if (iop.islower(tok[0])) err('Unknown language')
				err('Expected input')
		}
}

function parse(file, text) {
	var problem = {
		fns: new Map(),
		formulas: [],
	}
	parse1(file, text, null, problem)
	return problem
}

exports.parse = parse
