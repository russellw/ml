'use strict'
const path = require('path')
const assert = require('assert')
const logic = require('./logic')
const etc = require('./etc')
const cnf = require('./cnf')
const fs = require('fs')

var eof = ''

function unquote(s) {
	assert(s[0] === s[s.length - 1])
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
				// console.log(text.slice(toki, ti))
				continue
			}

			// block comment
			if (text.slice(ti, ti + 2) === '/*') {
				for (ti += 2; text.slice(ti, ti + 2) !== '*/'; ti++) if (ti === text.length) err("Unclosed '/*'")
				continue
			}

			// number
			if (/^[\+\-]?\d/.test(text.slice(ti, ti + 2))) {
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
					if (/^[\+\-]?[Ee]/.test(text.slice(ti, ti + 2))) {
						if (/[\+\-]/.test(text[ti])) ti++
						if (/[Ee]/.test(text[ti])) ti++
						while (/\d/.test(text[ti])) ti++
					}
				}
				tok = text.slice(toki, ti)
				return
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

	function id() {
		if (/^[a-z_]/.test(tok)) {
			var name = tok
			lex()
			return name
		}
		if (tok[0] === "'") {
			var name = unquote(tok)
			lex()
			return name
		}
		err('Expected name')
	}

	// types
	var types = new Map()

	function atomictype() {
		switch (tok) {
			case '!':
			case '[':
			case '$tType':
				throw 'Inappropriate'
			case '(':
				lex()
				var t = atomictype()
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
		var name = id()
		return etc.getor(types, name, () => {
			return { name }
		})
	}

	function topleveltype() {
		if (eat('(')) {
			var t = []
			do t.push(atomictype())
			while (eat('*'))
			expect(')')
			expect('>')
			t.unshift(atomictype())
			return t
		}
		var t = atomictype()
		if (eat('>')) return [atomictype(), t]
		return t
	}

	// terms
	var free = new Map()

	function args(bound, a = []) {
		assert(bound instanceof Map)
		expect('(')
		do a.push(term(bound))
		while (eat(','))
		expect(')')
		return a
	}

	function defined(bound, op, arity) {
		assert(bound instanceof Map)
		lex()
		var a = args(bound)
		if (a.length !== arity) err('Expected ' + arity + ' arguments')
		return logic.term(op, ...a)
	}

	function term(bound) {
		assert(bound instanceof Map)

		// defined functor
		switch (tok) {
			case '$difference':
				return defined(bound, '-', 2)
			case '$distinct':
				var a = args(bound)
				var clauses = logic.term('&')
				for (var i = 0; i < a.length; i++) for (var j = 0; j < i; j++) clauses.push(logic.term('!=', a[i], a[j]))
				return clauses
			case '$false':
				lex()
				return false
			case '$greater':
				return defined(bound, '>', 2)
			case '$greatereq':
				return defined(bound, '>=', 2)
			case '$less':
				return defined(bound, '<', 2)
			case '$lesseq':
				return defined(bound, '<=', 2)
			case '$product':
				return defined(bound, '*', 2)
			case '$quotient':
				return defined(bound, '/', 2)
			case '$sum':
				return defined(bound, '+', 2)
			case '$true':
				lex()
				return true
			case '$uminus':
				return defined(bound, 'unary-', 1)
			case '$quotient_e':
				return defined(bound, 'dive', 2)
			case '$quotient_f':
				return defined(bound, 'divf', 2)
			case '$quotient_t':
				return defined(bound, 'divt', 2)
			case '$remainder_e':
				return defined(bound, 'reme', 2)
			case '$remainder_f':
				return defined(bound, 'remf', 2)
			case '$remainder_t':
				return defined(bound, 'remt', 2)
			case '$is_int':
				return defined(bound, 'isinteger', 1)
			case '$is_rat':
				return defined(bound, 'isrational', 1)
			case '$to_int':
				return defined(bound, 'tointeger', 1)
			case '$to_rat':
				return defined(bound, 'torational', 1)
			case '$to_real':
				return defined(bound, 'toreal', 1)
			case '$ceiling':
				return defined(bound, 'ceil', 1)
			case '$floor':
				return defined(bound, 'floor', 1)
			case '$round':
				return defined(bound, 'round', 1)
			case '$truncate':
				return defined(bound, 'trunc', 1)
		}

		// integer
		if (/^[\+\-]?\d+$/.test(tok)) {
			var a = BigInt(tok)
			lex()
			return a
		}

		// rational or real
		if (/^[\+\-]?\d/.test(tok)) throw 'Inappropriate'

		// variable
		if (/^[A-Z]/.test(tok)) {
			var name = tok
			lex()
			if (bound.has(name)) return bound.get(name)
			if (!free) err('Unbound variable')
			return etc.getor(free, name, () => {
				return { op: 'var', type: 'individual' }
			})
		}

		// word
		if (/^[\w_]/.test(tok) || tok[0] === "'") {
			var name = id()
			var a = etc.getor(problem.fns, name, () => {
				return { name }
			})
			if (tok !== '(') return a
			return args(bound, logic.term('call', a))
		}

		// distinct object
		if (tok[0] === '"') {
			var s = unquote(tok)
			lex()
			return s
		}

		// other
		err('Expected term')
	}

	// formulas
	function eq(bound) {
		assert(bound instanceof Map)
		var a = term(bound)
		switch (tok) {
			case '!=':
				lex()
				var b = term(bound)
				return logic.term('!=', a, b)
			case '=':
				lex()
				var b = term(bound)
				return logic.term('==', a, b)
		}
		return a
	}

	function quant(bound, op) {
		assert(bound instanceof Map)
		lex()
		expect('[')
		bound = new Map(bound)
		var v = []
		do {
			var name = tok
			lex()
			var type = 'individual'
			if (eat(':')) type = atomictype()
			var x = { op: 'var', type }
			bound.set(name, x)
			v.push(x)
		} while (eat(','))
		expect(']')
		expect(':')
		return logic.term(op, v, unitary(bound))
	}

	function unitary(bound) {
		assert(bound instanceof Map)
		switch (tok) {
			case '!':
				return quant(bound, 'all')
			case '?':
				return quant(bound, 'exists')
			case '(':
				lex()
				var a = formula(bound)
				expect(')')
				return a
			case '~':
				lex()
				return logic.term('!', unitary(bound))
		}
		return eq(bound)
	}

	function formula(bound) {
		assert(bound instanceof Map)
		var a = unitary(bound)
		switch (tok) {
			case '&':
			case '|':
				var k = tok
				a = logic.term(k + k, a)
				while (eat(k)) a.push(unitary(bound))
				break
			case '<=':
				lex()
				return logic.term('=>', unitary(bound), a)
			case '<=>':
				lex()
				return logic.term('<=>', a, unitary(bound))
			case '<~>':
				lex()
				return logic.term('!', logic.term('<=>', a, unitary(bound)))
			case '=>':
				lex()
				return logic.term('=>', a, unitary(bound))
			case '~&':
				lex()
				return logic.term('!', logic.term('&&', a, unitary(bound)))
			case '~|':
				lex()
				return logic.term('!', logic.term('||', a, unitary(bound)))
		}
		return a
	}

	// top level
	function select(name) {
		if (!selection) return true
		return selection.has(name)
	}

	function ignore() {
		switch (tok) {
			case '(':
				lex()
				while (!eat(')')) ignore()
				return
			case eof:
				err("Expected ')'")
		}
		lex()
	}

	while (tok)
		switch (tok) {
			case 'cnf':
				lex()
				expect('(')

				// name
				var name = id()
				expect(',')

				// role
				id()
				expect(',')

				// literals
				var parens = eat('(')
				free = new Map()
				var neg = []
				var pos = []
				do {
					var no = eat('~')
					var a = eq(new Map())
					if (a.op === '!') {
						no = !no
						a = a[0]
					}
					;(no ? neg : pos).push(a)
				} while (eat('|'))
				if (parens) expect(')')

				// select
				if (select(name)) {
					var c = [neg, pos]
					c.file = file
					problem.clauses.push(c)
				}

				// annotations
				if (tok === ',') while (tok !== ')') ignore()

				// end
				expect(')')
				expect('.')
				break
			case 'fof':
			case 'tff':
				lex()
				expect('(')

				// name
				var name = id()
				expect(',')

				// role
				var role = id()
				expect(',')

				if (role === 'type') {
					// type declaration
					var parens = 0
					while (eat('(')) parens++

					var name = id()
					expect(':')
					if (eat('$tType')) {
						if (tok === '>') throw 'Inappropriate'
					} else {
						var a = etc.getor(problem.fns, name, () => {
							return { name }
						})
						var toki1 = toki
						var type = topleveltype()
						if (!a.type) a.type = type
						else if (!logic.eq(a.type, type)) {
							toki = toki1
							err('Type mismatch')
						}
					}

					while (parens--) expect(')')
				} else {
					// formula
					free = null
					var a = formula(new Map())
					if (role === 'conjecture') {
						if (problem.conjecture) err('Multiple conjectures not supported')
						problem.conjecture = a
						a = logic.term('!', a)
					}

					// select
					if (select(name)) problem.formulas.push(a)
				}

				// annotations
				if (tok === ',') while (tok !== ')') ignore()

				// end
				expect(')')
				expect('.')
				break
			case 'include':
				lex()
				expect('(')

				// file
				var name = id()

				// selection
				var selection1 = selection
				if (eat(',')) {
					expect('[')
					selection1 = new Set()
					do {
						var s = id()
						if (select(s)) selection1.add(s)
					} while (eat(','))
					expect(']')
				}

				// end
				expect(')')
				expect('.')

				// include
				if (!path.isAbsolute(name)) {
					var tptp = process.env.TPTP
					if (!tptp) err('TPTP environment variable not defined')
					name = tptp + '/' + name
				}
				var text1 = fs.readFileSync(name, 'utf8')
				parse1(name, text1, selection1, problem)
				break
			default:
				err('Syntax error')
		}
}

function parse(file, text) {
	var problem = {
		fns: new Map(),
		formulas: [],
		clauses: [],
	}
	parse1(file, text, null, problem)
	return problem
}

assert(/^[\+\-]?\d/.test('9'))
assert(/^[\+\-]?\d/.test('99'))
assert(/^[\+\-]?\d/.test('9x'))
assert(/^[\+\-]?\d/.test('+9'))
assert(/^[\+\-]?\d/.test('-9'))
assert(!/^[\+\-]?\d/.test('x'))
assert(!/^[\+\-]?\d/.test('x9'))

assert(BigInt('3') === 3n)
assert(BigInt('+3') === 3n)
assert(BigInt('-3') === -3n)

exports.parse = parse
