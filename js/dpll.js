'use strict'
var logic = require('./logic')
var cnf = require('./cnf')

function sat(clauses, m = new Map()) {
	var cs=cnf.simplifyClauses(clauses)
	if (cnf.isFalse(clauses)) return
	if (cnf.isTrue(clauses)) return m

	// Unit clauses
	for (var clause of clauses)
		if (clause.length === 1) {
			var literal = clause[0]
			var polarity = literal.op !== '~'
			var atom = polarity ? literal : literal[0]
			return sat(cnf.evaluate(clauses, cnf.empty.add(atom, cnf.bool(polarity))))
		}

	// Atoms
	var atoms = cnf.empty
	for (var clause of clauses)
		for (var literal of clause) {
			var polarity = literal.op !== '~'
			var atom = polarity ? literal : literal[0]
			atoms = atoms.add(atom, true)
		}
	atoms = atoms.keys

	function occurs(polarity1, atom1) {
		for (var clause of clauses)
			for (var literal of clause) {
				var polarity = literal.op !== '~'
				var atom = polarity ? literal : literal[0]
				if (polarity === polarity1 && cnf.eq(atom, atom1)) return true
			}
	}

	// Pure atoms
	for (var clause of clauses)
		for (var literal of clause) {
			var polarity = literal.op !== '~'
			var atom = polarity ? literal : literal[0]
			if (!occurs(!polarity, atom)) return sat(cnf.evaluate(clauses, cnf.empty.add(atom, cnf.bool(polarity))))
		}

	// Guess
	var atom = atoms[0]
	var r = sat(cnf.evaluate(clauses, cnf.empty.add(atom, cnf.bool(false))))
	if (r) return r
	return sat(cnf.evaluate(clauses, cnf.empty.add(atom, cnf.bool(true))))
}

exports.sat = sat
