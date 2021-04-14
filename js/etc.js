'use strict'

function walk(a, f) {
	f(a)
	if (Array.isArray(a)) for (var b of a) walk(b, f)
}

function err(file, text, toki, msg) {
	// line number
	var line = 1
	for (var i = 0; i < toki; i++) if (text[i] === '\n') line++

	// start of line
	var linestart = toki
	while (linestart && text[linestart - 1] !== '\n') linestart--

	// print context
	for (var i = linestart; text[i] >= ' ' || text[i] === '\t'; i++);
	console.error(text.slice(linestart, i))

	// print caret
	for (var i = linestart; i < toki; i++) process.stderr.write(text[i] === '\t' ? '\t' : ' ')
	console.error('^')

	// print message and exit
	console.error('%s:%d: %s', file, line, msg)
	process.exit(1)
}

exports.err = err
exports.walk = walk
