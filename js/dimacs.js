'use strict'
var logic = require('./logic')
var etc = require('./etc')

// Tokenizer
var file
var ti
var status
var text
var tok
var tokstart

function err(msg) {
	etc.err(file, text, tokstart, msg)
}
