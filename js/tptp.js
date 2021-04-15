'use strict'
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
	var types = new Map()

	function atomictype() {
		switch (tok) {
			case '!':
			case '[':
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
		if (/^[\w_]+/.test(tok)) {
			var name = tok
			lex()
			return etc.getor(types, name, () => {
				return { name }
			})
		}
		if (tok[0] === "'") {
			var name = unquote(tok)
			lex()
			return etc.getor(types, name, () => {
				return { name }
			})
		}
		err('Expected type')
	}

	function topleveltype() {
		if (eat(')')) {
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
		expect('(')
		do a.push(term(bound))
		while (eat(','))
		expect(')')
		return a
	}

	function defined(bound, op, arity) {
		var a = args(bound)
		if (a.length !== arity) err('Expected ' + arity + ' arguments')
		return cnf.term(op, ...a)
	}

	function plain(bound, name) {
		lex()
		var a = etc.getor(problem.fns, name, () => {
			return { name }
		})
		if (tok !== '(') return a
		var a = args(bound)
		return cnf.call(f, a)
	}

	function term(bound) {
		switch (tok) {
			case '$difference':
				return defined(bound, '-', 2)
			case '$distinct':
				var a = args(bound)
				var clauses = cnf.term('&')
				for (var i = 0; i < a.length; i++) for (var j = 0; j < i; j++) clauses.push(cnf.term('!=', a[i], a[j]))
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
		}
		switch (tok[0]) {
			case '"':
				var s = unquote(tok)
				lex()
				return s
			case "'":
				return plain(unquote(tok))
		}

		// word
		if (/^[a-z_]/.test(tok)) return plain(tok)

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

		// number
		if (/^[\+\-]?\d+$/.test(tok)) {
			var a = BigInt(tok)
			lex()
			return a
		}
		if (/^[\+\-]?\d/.test(tok)) throw 'Inappropriate'

		// other
		err('Expected term')
	}

	// formulas
	function infix_unary(bound) {
		var a = term(bound)
		switch (tok) {
			case '!=':
				lex()
				var b = term()
				return cnf.term('!=', a, b)
			case '=':
				lex()
				var b = term()
				return cnf.term('==', a, b)
		}
		return a
	}

	function quant(bound) {
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
		return logic.term(op, v, unitary_formula(bound))
	}

	function unitary_formula(bound) {
		switch (tok) {
			case '!':
				return quant('all')
			case '?':
				return quant('exists')
			case '(':
				lex()
				var a = formula(bound)
				expect(')')
				return a
			case '~':
				lex()
				return cnf.term('!', unitary_formula(bound))
		}
		return infix_unary(bound)
	}

	function formula(bound) {
		var a = unitary_formula(bound)
		switch (tok) {
			case '&':
			case '|':
				var k = tok
				a = logic.term(k + k, a)
				while (eat(k)) a.push(unitary_formula(bound))
				break
			case '<=':
				lex()
				return logic.term('=>', unitary_formula(bound), a)
			case '<=>':
				lex()
				return logic.term('<=>', a, unitary_formula(bound))
			case '<~>':
				lex()
				return logic.term('!', logic.term('<=>', a, unitary_formula(bound)))
			case '=>':
				lex()
				return logic.term('=>', a, unitary_formula(bound))
			case '~&':
				lex()
				return logic.term('!', logic.term('&&', a, unitary_formula(bound)))
			case '~|':
				lex()
				return logic.term('!', logic.term('||', a, unitary_formula(bound)))
		}
		return a
	}

	// top level
	function formula_name() {
		if (/^\w/.test(tok)) {
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
				var name = formula_name()
				expect(',')

				// role
				if (!/^[a-z]/.test(tok)) err('Expected role')
				var role = tok
				lex()
				expect(',')

				// literals
				free = new Map()

				// annotations
				if (eat(',')) while (tok !== ')') ignore()

				// end
				expect(')')
				expect('.')
				break
			case 'fof':
			case 'tff':
				lex()
				expect('(')

				// name
				var name = formula_name()
				expect(',')

				// role
				if (!/^[a-z]/.test(tok)) err('Expected role')
				var role = tok
				lex()
				expect(',')

				// formula
				free = null
				var a = formula(new Map())
				if (role === 'conjecture') {
					if (problem.conjecture) err('Multiple conjectures not supported')
					problem.conjecture = a
					a = cnf.term('!', a)
				}
				if (select(name)) problem.formulas.push(a)

				// annotations
				if (eat(',')) while (tok !== ')') ignore()

				// end
				expect(')')
				expect('.')
				break
			case 'include':
				lex()
				expect('(')

				// file
				if (tok[0] !== "'") err('Expected file')
				var name = unquote(tok)
				lex()

				// selection
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
				parse1(name, text1, selection1)
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
