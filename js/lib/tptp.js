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

function parse1(file, txt, selection, problem) {
	var ti = 0
	var tok = null

	function err(msg) {
		console.error('%s:%d: %s', file, txt.slice(0, ti).split('\n').length, msg)
		process.exit(1)
	}

	// tokenizer
	function lex() {
		while (ti < txt.length) {
			var ti0 = ti

			// space
			if (/\s/.test(txt[ti])) {
				ti++
				continue
			}

			// line comment
			if (txt[ti] === '%') {
				while (txt[ti] !== '\n') ti++
				var s = txt.slice(ti0, ti)
				if (!problem.doneheader) console.log(s)
				if (!problem.expected) {
					var m = /%\s*Status\s*:\s*(\w+)/.exec(s)
					if (m) problem.expected = m[1]
				}
				continue
			}

			// block comment
			if (txt.slice(ti, ti + 2) === '/*') {
				for (ti += 2; txt.slice(ti, ti + 2) !== '*/'; ti++)
					if (ti === txt.length) {
						ti = ti0
						err("unclosed '/*'")
					}
				ti += 2
				continue
			}

			// number
			if (/^[\+\-]?\d/.test(txt.slice(ti, ti + 2))) {
				// sign
				if (/[\+\-]/.test(txt[ti])) ti++

				// integer
				while (/\d/.test(txt[ti])) ti++

				// fraction
				if (txt[ti] === '/') {
					// denominator
					ti++
					while (/\d/.test(txt[ti])) ti++
				} else {
					// decimal
					if (txt[ti] === '.') {
						ti++
						while (/\d/.test(txt[ti])) ti++
					}

					// exponent
					if (/^[\+\-]?[Ee]/.test(txt.slice(ti, ti + 2))) {
						if (/[\+\-]/.test(txt[ti])) ti++
						if (/[Ee]/.test(txt[ti])) ti++
						while (/\d/.test(txt[ti])) ti++
					}
				}
				tok = txt.slice(ti0, ti)
				return
			}

			// word
			if (/[\w_\$]/.test(txt[ti])) {
				while (/[\w_\$]/.test(txt[ti])) ti++
				tok = txt.slice(ti0, ti)
				return
			}

			// quote
			if (txt[ti] === "'" || txt[ti] === '"') {
				for (var q = txt[ti++]; txt[ti] !== q; ti++) {
					if (txt[ti] === '\\') ti++
					if (ti === txt.length) {
						ti = ti0
						err('unclosed quote')
					}
				}
				ti++
				tok = txt.slice(ti0, ti)
				return
			}

			// 3-char punctuation
			var punct = ['<=>', '<~>']
			if (punct.includes(txt.slice(ti, ti + 3))) {
				ti += 3
				tok = txt.slice(ti0, ti)
				return
			}

			// 2-char punctuation
			var punct = ['!=', '=>', '<=', '~|', '~&']
			if (punct.includes(txt.slice(ti, ti + 2))) {
				ti += 2
				tok = txt.slice(ti0, ti)
				return
			}

			// other
			tok = txt[ti++]
			return
		}
		tok = eof
	}

	lex()
	problem.doneheader = true

	function eat(k) {
		if (tok === k) {
			lex()
			return true
		}
	}

	function expect(k) {
		if (!eat(k)) err("expected '" + k + "'")
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
		err('expected name')
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

	function requiretype(a, t) {
		etc.defaulttype(a, t)
		if (!etc.eq(etc.type(a), t)) err('type mismatch')
	}

	// terms
	var free = new Map()

	function args(env) {
		assert(env instanceof Map)
		expect('(')
		var a = []
		do a.push(term(env))
		while (eat(','))
		expect(')')
		return a
	}

	function defined(env, o, arity) {
		assert(env instanceof Map)
		lex()
		var a = args(env)
		if (a.length !== arity) err('expected ' + arity + ' arguments')
		if (!etc.isnumtype(etc.type(a[0]))) err('expected numeric term')
		for (var i = 1; i < a.length; i++) requiretype(a[i], etc.type(a[0]))
		return etc.mk(o, ...a)
	}

	function term(env) {
		assert(env instanceof Map)

		// defined functor
		switch (tok) {
			case '!':
			case '$ite':
				throw 'Inappropriate'
			case '$difference':
				return defined(env, '-', 2)
			case '$distinct':
				lex()
				var a = args(env)
				etc.defaulttype(a[0], 'individual')
				for (var i = 1; i < a.length; i++) requiretype(a[i], etc.type(a[0]))
				var inequalities = etc.mk('&')
				for (var i = 0; i < a.length; i++)
					for (var j = 0; j < i; j++) inequalities.push(etc.mk('!', etc.mk('==', a[i], a[j])))
				return inequalities
			case '$false':
				lex()
				return false
			case '$greater':
				var a = defined(env, '>', 2)
				return etc.mk('<', a[1], a[0])
			case '$greatereq':
				var a = defined(env, '>=', 2)
				return etc.mk('<=', a[1], a[0])
			case '$less':
				return defined(env, '<', 2)
			case '$lesseq':
				return defined(env, '<=', 2)
			case '$product':
				return defined(env, '*', 2)
			case '$quotient':
				var a = defined(env, '/', 2)
				if (type(a[0]) === 'bigint') err('expected rational or real')
				return a
			case '$sum':
				return defined(env, '+', 2)
			case '$true':
				lex()
				return true
			case '$uminus':
				return defined(env, 'unary-', 1)
			case '$quotient_e':
				return defined(env, 'dive', 2)
			case '$quotient_f':
				return defined(env, 'divf', 2)
			case '$quotient_t':
				return defined(env, 'divt', 2)
			case '$remainder_e':
				return defined(env, 'reme', 2)
			case '$remainder_f':
				return defined(env, 'remf', 2)
			case '$remainder_t':
				return defined(env, 'remt', 2)
			case '$is_int':
				return defined(env, 'isint', 1)
			case '$is_rat':
				return defined(env, 'israt', 1)
			case '$to_int':
				return defined(env, 'toint', 1)
			case '$to_rat':
				return defined(env, 'torat', 1)
			case '$to_real':
				return defined(env, 'toreal', 1)
			case '$ceiling':
				return defined(env, 'ceil', 1)
			case '$floor':
				return defined(env, 'floor', 1)
			case '$round':
				return defined(env, 'round', 1)
			case '$truncate':
				return defined(env, 'trunc', 1)
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
			if (env.has(name)) return env.get(name)
			if (!free) err(name + ': unbound variable')
			return etc.getor(free, name, () => {
				return { o: 'var', type: 'individual' }
			})
		}

		// word
		if (/^[\w_]/.test(tok) || tok[0] === "'") {
			var name = id()
			var f = etc.getor(problem.fns, name, () => {
				return { o: 'fn', name }
			})
			if (tok !== '(') return f
			var a = args(env)
			for (var b of a) {
				etc.defaulttype(b, 'individual')
				if (etc.type(b) === 'boolean') err('term cannot be boolean')
			}
			return etc.mk('call', ...[f].concat(a))
		}

		// distinct object
		if (tok[0] === '"') {
			var s = tok
			lex()
			return s
		}

		// other
		err('expected term')
	}

	// formulas
	function eq(env) {
		assert(env instanceof Map)
		var a = term(env)
		switch (tok) {
			case '!=':
				lex()
				var b = term(env)
				etc.defaulttype(a, 'individual')
				requiretype(b, etc.type(a))
				return etc.mk('!', etc.mk('==', a, b))
			case '=':
				lex()
				var b = term(env)
				etc.defaulttype(a, 'individual')
				requiretype(b, etc.type(a))
				return etc.mk('==', a, b)
		}
		requiretype(a, 'boolean')
		return a
	}

	function quant(env, o) {
		assert(env instanceof Map)
		lex()
		expect('[')
		env = new Map(env)
		var v = []
		do {
			var name = tok
			lex()
			var x = { o: 'var', type: 'individual' }
			if (eat(':')) x.type = atomictype()
			env.set(name, x)
			v.push(x)
		} while (eat(','))
		expect(']')
		expect(':')
		return etc.mk(o, v, unitary(env))
	}

	function unitary(env) {
		assert(env instanceof Map)
		switch (tok) {
			case '!':
				return quant(env, 'all')
			case '?':
				return quant(env, 'exists')
			case '(':
				lex()
				var a = formula(env)
				expect(')')
				return a
			case '~':
				lex()
				return etc.mk('!', unitary(env))
		}
		return eq(env)
	}

	function formula(env) {
		assert(env instanceof Map)
		var a = unitary(env)
		switch (tok) {
			case '&':
			case '|':
				var k = tok
				a = etc.mk(k + k, a)
				while (eat(k)) a.push(unitary(env))
				break
			case '<=':
				lex()
				return etc.mk('||', a, etc.mk('!', unitary(env)))
			case '<=>':
				lex()
				return etc.mk('<=>', a, unitary(env))
			case '<~>':
				lex()
				return etc.mk('!', etc.mk('<=>', a, unitary(env)))
			case '=>':
				lex()
				return etc.mk('||', etc.mk('!', a), unitary(env))
			case '~&':
				lex()
				return etc.mk('!', etc.mk('&&', a, unitary(env)))
			case '~|':
				lex()
				return etc.mk('!', etc.mk('||', a, unitary(env)))
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
				err("expected ')'")
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
					c.file = path.basename(file)
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
							return { o: 'fn', name }
						})
						requiretype(a, topleveltype())
					}

					while (parens--) expect(')')
				} else {
					// formula
					free = null
					var a = formula(new Map())
					var c = [a]
					c.file = path.basename(file)
					c.name = name

					// negate conjecture
					if (role === 'conjecture') {
						if (problem.conjecture) err('multiple conjectures are ambiguous')
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
				err('syntax error')
		}
}

function parse(file, txt) {
	var problem = {
		types: new Map(),
		fns: new Map(),
		formulas: [],
		clauses: [],
	}
	parse1(file, txt, null, problem)
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

var debugnames = 0

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
		case '&&':
			infix(' & ', a, parent)
			return
		case '||':
			infix(' | ', a, parent)
			return
		case '<=>':
			infix(' ' + a.o + ' ', a, parent)
			return
		case 'all':
			process.stdout.write('!')
			quant(a)
			return
		case 'exists':
			process.stdout.write('?')
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
		case 'isint':
			process.stdout.write('$is_int')
			args(a)
			return
		case 'israt':
			process.stdout.write('$is_rat')
			args(a)
			return
		case 'toint':
			process.stdout.write('$to_int')
			args(a)
			return
		case 'torat':
			process.stdout.write('$to_rat')
			args(a)
			return
		case 'toreal':
			process.stdout.write('$to_real')
			args(a)
			return
		case 'ceil':
			process.stdout.write('$ceiling')
			args(a)
			return
		case 'floor':
			process.stdout.write('$floor')
			args(a)
			return
		case 'round':
			process.stdout.write('$round')
			args(a)
			return
		case 'trunc':
			process.stdout.write('$truncate')
			args(a)
			return
		case '<':
			process.stdout.write('$less')
			args(a)
			return
		case '<=':
			process.stdout.write('$lesseq')
			args(a)
			return
		case '-':
			process.stdout.write('$difference')
			args(a)
			return
		case '*':
			process.stdout.write('$product')
			args(a)
			return
		case '/':
			process.stdout.write('$quotient')
			args(a)
			return
		case 'dive':
			process.stdout.write('$quotient_e')
			args(a)
			return
		case 'divt':
			process.stdout.write('$quotient_t')
			args(a)
			return
		case 'divf':
			process.stdout.write('$quotient_f')
			args(a)
			return
		case 'unary-':
			process.stdout.write('$uminus')
			args(a)
			return
		case 'reme':
			process.stdout.write('$remainder_e')
			args(a)
			return
		case 'remt':
			process.stdout.write('$remainder_t')
			args(a)
			return
		case 'remf':
			process.stdout.write('$remainder_f')
			args(a)
			return
		case 'call':
			prterm(a[0])
			args(a.slice(1))
			return
		case '!':
			if (a[0].o === '==') {
				infix('!=', a[0])
				return
			}
			process.stdout.write('~')
			prterm(a[0])
			return
		case 'fn':
			if (!a.name) a.name = debugnames++
			if (typeof a.name === 'number') {
				process.stdout.write('#' + a.name.toString(16))
				return
			}
			prname(a.name)
			return
	}
	etc.show(a)
	assert(false)
}

function prnterm(a) {
	prterm(a)
	console.log()
}

function prclausename(c) {
	process.stdout.write(String(c.name))
}

function prnclause(c) {
	varnames = new Map()
	process.stdout.write(c.length === 2 ? 'cnf(' : 'fof(')
	prclausename(c)
	process.stdout.write(', ')

	// role
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
	process.stdout.write(', ')

	// term
	if (c.length === 2) {
		var [neg, pos] = c
		for (var i = 0; i < neg.length; i++) {
			if (i) process.stdout.write(' | ')
			prterm(etc.mk('!', neg[i]))
		}
		for (var i = 0; i < pos.length; i++) {
			if (neg.length + i) process.stdout.write(' | ')
			prterm(pos[i])
		}
		if (neg.length + pos.length === 0) process.stdout.write('$false')
	} else prterm(c[0])
	process.stdout.write(', ')

	// source
	switch (c.how) {
		case 'negate':
			process.stdout.write('inference(negate,[status(ceq)],[')
			prclausename(c.from[0])
			process.stdout.write('])')
			break
		default:
			if (c.file) {
				process.stdout.write('file(' + etc.quote("'", c.file) + ',')
				prclausename(c)
				process.stdout.write(')')
				break
			}
			process.stdout.write('inference(' + c.how + ',[status(')
			process.stdout.write(c.how === 'cnf' ? 'esa' : 'thm')
			process.stdout.write(')]')
			if (c.from) {
				process.stdout.write(',[')
				for (var i = 0; i < c.from.length; i++) {
					if (i) process.stdout.write(',')
					prclausename(c.from[i])
				}
				process.stdout.write(']')
			}
			process.stdout.write(')')
			break
	}
	console.log(').')
}

function walkproof(c, proof, visited = new Set()) {
	if (visited.has(c)) return
	visited.add(c)
	if (c.from) for (var d of c.from) walkproof(d, proof, visited)
	proof.push(c)
}

function prnproof(conclusion) {
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
			if (a.o !== 'fn') return
			var m = /^_sK(\d+)$/.exec(a.name)
			if (m) i = Math.max(i, parseInt(m[1], 10))
		})
	i++

	// name Skolem functions that were not already named
	for (var c of proof)
		etc.walk(c, (a) => {
			if (a.o !== 'fn') return
			if (!a.name) a.name = 'sK' + String(i++)
		})

	// print clauses
	for (var c of proof) prnclause(c)
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

module.exports = {
	prnterm,
	prnproof,
	prnclause,
	parse,
}
