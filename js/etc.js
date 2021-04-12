'use strict'

function err(file, text, tokstart, msg) {
	// line number
	var line = 1
	for (var i = 0; i < tokstart; i++) if (text[i] == '\n') line++

	// start of line
	var linestart = tokstart
	while (linestart && text[linestart - 1] !== '\n') linestart--

	// print context
	for (var i = linestart; text[i] >= ' ' || text[i] == '\t'; i++);
	console.error(text.slice(linestart, i))

	// print caret
	for (var i = linestart; i != tokstart; i++) process.stderr.write(text[i] == '\t' ? '\t' : ' ')
	console.error('^')

	// print message and exit
	console.error('%s:%d: %s', file, line, msg)
	process.exit(1)
}

exports.err = err
