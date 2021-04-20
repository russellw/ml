'use strict'
const path = require('path')
const assert = require('assert')
const etc = require('./etc')
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
				ti += 2
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
		if (/^[\w_]/.test(tok)) {
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
				return 'boolean'
			case '$int':
				lex()
				return 'bigint'
			case '$rat':
				lex()
				return 'rat'
			case '$real':
				lex()
				return 'real'
		}
		var name = id()
		return etc.getor(problem.types, name, () => {
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

	function defined(bound, o, arity) {
		assert(bound instanceof Map)
		lex()
		var a = args(bound)
		if (a.length !== arity) err('Expected ' + arity + ' arguments')
		return etc.mk(o, ...a)
	}

	function term(bound) {
		assert(bound instanceof Map)

		// defined functor
		switch (tok) {
			case '!':
			case '$ite':
				throw 'Inappropriate'
			case '$difference':
				return defined(bound, '-', 2)
			case '$distinct':
				lex()
				var a = args(bound)
				var clauses = etc.mk('&')
				for (var i = 0; i < a.length; i++) for (var j = 0; j < i; j++) clauses.push(etc.mk('!=', a[i], a[j]))
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
				return defined(bound, 'isint', 1)
			case '$is_rat':
				return defined(bound, 'israt', 1)
			case '$to_int':
				return defined(bound, 'toint', 1)
			case '$to_rat':
				return defined(bound, 'torat', 1)
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
				return { o: 'var', type: 'individual' }
			})
		}

		// word
		if (/^[\w_]/.test(tok) || tok[0] === "'") {
			var name = id()
			var a = etc.getor(problem.fns, name, () => {
				return { name }
			})
			if (tok !== '(') return a
			return args(bound, etc.mk('call', a))
		}

		// distinct object
		if (tok[0] === '"') {
			var s = tok
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
				return etc.mk('!=', a, b)
			case '=':
				lex()
				var b = term(bound)
				return etc.mk('==', a, b)
		}
		return a
	}

	function quant(bound, o) {
		assert(bound instanceof Map)
		lex()
		expect('[')
		bound = new Map(bound)
		var v = []
		do {
			var name = tok
			lex()
			var t = 'individual'
			if (eat(':')) t = atomictype()
			var x = { o: 'var', type: t }
			bound.set(name, x)
			v.push(x)
		} while (eat(','))
		expect(']')
		expect(':')
		return etc.mk(o, v, unitary(bound))
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
				return etc.mk('!', unitary(bound))
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
				a = etc.mk(k + k, a)
				while (eat(k)) a.push(unitary(bound))
				break
			case '<=':
				lex()
				return etc.mk('=>', unitary(bound), a)
			case '<=>':
				lex()
				return etc.mk('<=>', a, unitary(bound))
			case '<~>':
				lex()
				return etc.mk('!', etc.mk('<=>', a, unitary(bound)))
			case '=>':
				lex()
				return etc.mk('=>', a, unitary(bound))
			case '~&':
				lex()
				return etc.mk('!', etc.mk('&&', a, unitary(bound)))
			case '~|':
				lex()
				return etc.mk('!', etc.mk('||', a, unitary(bound)))
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
					if (a.o === '!') {
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
					c.name = name
					problem.clauses.push(c)
				}

				// annotations
				if (tok === ',') while (tok !== ')') ignore()

				// end
				expect(')')
				expect('.')
				break
			case 'thf':
				throw 'Inappropriate'
			case 'fof':
			case 'tff':
				lex()
				expect('(')

				// name
				var name = id()
				expect(',')

				// role
				if (tok === 'conjecture' && problem.conjecture) err('Multiple conjectures are ambiguous')
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
						var t = topleveltype()
						if (!a.type) a.type = t
						else if (!etc.eq(a.type, t)) {
							toki = toki1
							err('Type mismatch')
						}
					}

					while (parens--) expect(')')
				} else {
					// formula
					free = null
					var a = formula(new Map())
					var c = [a]
					c.file = file
					c.name = name

					// negate conjecture
					if (role === 'conjecture') {
						c.how = 'conjecture'
						problem.conjecture = c
						a = etc.mk('!', a)
						c = [a]
						c.how = 'negate'
						c.from = [problem.conjecture]
					}

					// select
					if (select(name)) problem.formulas.push(c)
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
		types: new Map(),
		fns: new Map(),
		formulas: [],
		clauses: [],
	}
	parse1(file, text, null, problem)
	return problem
}

function prname(s) {
	assert(s)
	process.stdout.write(/^[a-z][\w_]*$/.test(s) ? s : etc.quote("'", s))
}

function prtype(t) {
	switch (t) {
		case 'boolean':
			process.stdout.write('$o')
			return
		case 'individual':
			process.stdout.write('$i')
			return
		case 'bigint':
			process.stdout.write('$int')
			return
		case 'rat':
		case 'real':
			process.stdout.write('$' + t)
			return
	}
	assert(typeof t === 'object')
	if (!Array.isArray(t)) {
		prname(t.name)
		return
	}
	if (t.length > 2) process.stdout.write('(')
	for (var i = 1; i < t.length; i++) {
		if (i > 1) process.stdout.write(' * ')
		prterm(t[i])
	}
	if (t.length > 2) process.stdout.write(')')
	process.stdout.write(' * ')
	prtype(a[0])
}

var skolemname = 0
var varnames = new Map()

function args(a) {
	process.stdout.write('(')
	for (var i = 0; i < a.length; i++) {
		if (i) process.stdout.write(',')
		prterm(a[i])
	}
	process.stdout.write(')')
}

function quant(a) {
	process.stdout.write('[')
	for (var i = 0; i < a[0].length; i++) {
		if (i) process.stdout.write(',')
		var x = a[0][i]
		prterm(x)
		assert(x.type)
		if (x.type !== 'individual') {
			process.stdout.write(':')
			prtype(x.type)
		}
	}
	process.stdout.write(']:')
	prterm(a[1], a)
}

function needparens(a, parent) {
	switch (a.o) {
		case '&&':
		case '||':
		case '<=>':
		case '=>':
			if (parent)
				switch (parent.o) {
					case '&&':
					case '||':
					case '<=>':
					case '=>':
					case 'all':
					case 'exists':
					case '!':
						return true
						break
				}
			break
	}
}

function infix(k, a, parent) {
	if (needparens(a, parent)) process.stdout.write('(')
	for (var i = 0; i < a.length; i++) {
		if (i) process.stdout.write(k)
		prterm(a[i], a)
	}
	if (needparens(a, parent)) process.stdout.write(')')
}

function prterm(a, parent) {
	switch (typeof a) {
		case 'boolean':
			process.stdout.write('$')
		case 'bigint':
		case 'string':
			process.stdout.write(String(a))
			return
	}
	switch (a.o) {
		case '==':
			infix('=', a)
			return
		case '!=':
			infix('!=', a)
			return
		case '&&':
			infix(' & ', a, parent)
			return
		case '||':
			infix(' | ', a, parent)
			return
		case '<=>':
		case '=>':
			infix(' ' + a.o + ' ', a, parent)
			return
		case 'all':
			process.stdout.write('!')
			quant(a)
			return
		case 'var':
			if (!varnames.has(a)) {
				var i = varnames.size
				varnames.set(a, i < 26 ? String.fromCharCode(65 + i) : 'Z' + (i - 25))
			}
			process.stdout.write(varnames.get(a))
			return
		case '+':
			process.stdout.write('$sum')
			args(a)
			return
		case 'call':
			prterm(a[0])
			args(a.slice(1))
			return
		case '!':
			process.stdout.write('~')
			prterm(a[0])
			return
	}
	if (etc.isfn(a)) {
		prname(a.name)
		return
	}
	etc.show(a)
	assert(false)
}

function prclause(c) {
	varnames = new Map()
	process.stdout.write(c.length === 2 ? 'cnf(' : 'fof(')
	process.stdout.write(c.name)
	process.stdout.write(', ')

	// role
	process.stdout.write(', ')
	switch (c.how) {
		case 'negate':
			process.stdout.write('negated_conjecture')
			break
		case 'conjecture':
			process.stdout.write('conjecture')
			break
		default:
			process.stdout.write('plain')
			break
	}

	// term
	if (c.length === 2) {
		for (var i = 0; i < c[0].length; i++) {
			if (i) process.stdout.write(' | ')
			process.stdout.write('~')
			prterm(c[0][i])
		}
		for (var i = 0; i < c[1].length; i++) {
			if (c[0].length + i) process.stdout.write(' | ')
			prterm(c[1][i])
		}
	} else prterm(c[0])
	process.stdout.write(', ')

	// source
	switch (c.how) {
		case 'negate':
			process.stdout.write('inference(negate,[status(ceq)],[' + c.from[0].name + '])')
			break
		default:
			if (c.file) {
				process.stdout.write('file(' + etc.quote("'", c.file) + ',' + c.name + ')')
				break
			}
			process.stdout.write('inference(' + c.how + ',[status(')
			process.stdout.write(')],[')
			for (var i = 0; i < c.from.length; i++) {
				if (i) process.stdout.write(',')
				process.stdout.write(c.from[i].name)
			}
			process.stdout.write('])')
			break
	}
	console.log(').')
}

function walkproof(c, proof, visited = new Set()) {
	if (visited.has(c)) return
	visited.add(c)
	for (var d of c.from) walkproof(d, proof, visited)
	proof.push(c)
}

function prproof(conclusion) {
	var proof = []
	walkproof(conclusion, proof)

	// check highest number already used for clause name
	var i = 0
	for (var c of proof) if (/^\d+$/.test(c.name)) i = Math.max(i, parseInt(c.name, 10))
	i++

	// name clauses that were not already named
	for (var c of proof) if (!c.name) c.name = String(i++)

	// check highest number already used for Skolem function name
	var i = 0
	for (var c of proof)
		etc.walk(c, (a) => {
			if (!etc.isfn(a)) return
			var m = /^_sK(\d+)$/.match(a.name)
			if (m) i = Math.max(i, parseInt(m[1], 10))
		})
	i++

	// name Skolem functions that were not already named
	for (var c of proof)
		etc.walk(c, (a) => {
			if (!etc.isfn(a)) return
			if (!a.name) a.name = 'sK' + String(i++)
		})

	// print clauses
	for (var c of proof) prclause(c)
}

function test() {
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

	assert(/^[a-z][\w_]*$/.test('a'))
	assert(/^[a-z][\w_]*$/.test('a9'))
	assert(/^[a-z][\w_]*$/.test('aA'))
	assert(/^[a-z][\w_]*$/.test('a_'))
	assert(!/^[a-z][\w_]*$/.test('9'))
	assert(!/^[a-z][\w_]*$/.test('A'))
	assert(!/^[a-z][\w_]*$/.test('$foo'))
}

test()

exports.parse = parse
exports.prclause = prclause
exports.prproof = prproof
