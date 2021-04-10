'use strict'
var assert = require('assert')

function occurs(a, b, m) {
    if (a === b) return true
    if (m.has(b)) return occurs(a, m.get(b), m)
    if (!b.args) return false
    for (var x of b.args) if (occurs(a, x, m)) return true
}

function unify(a, b, m = new Map()) {
    if (a === b) return m
    if (a.op === 'var') return unifyVar(a, b, m)
    if (b.op === 'var') return unifyVar(b, a, m)
    if (a.op !== b.op) return null
    switch (a.op) {
        case 'call':
            if (a.fun !== b.fun) return null
            break
        case 'const':
            return a.val === b.val
    }
    if (!a.args) return m
    if (a.args.length !== b.args.length) return null
    for (var i = 0; i < a.args.length && m; i++)
        m = unify(a.args[i], b.args[i], m)
    return m
}

function unifyVar(a, b, m) {
    if (m.has(a)) return unify(m.get(a), b, m)
    if (m.has(b)) return unify(a, m.get(b), m)
    if (occurs(a, b, m)) return null
    m.set(a, b)
    return m
}

exports.occurs = occurs
exports.unify = unify
