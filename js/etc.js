'use strict'

function err(file,text,tokstart,msg) {
  // line number
  var line = 1;
  for(var i=0;i<tokstart;i++)
    if (text[i] == '\n')
      line++;

  // start of line
  var linestart = tokstart;
  while (linestart && text[linestart-1] !== '\n')
    linestart--;

  // print context
  for (var i = linestart; text[i] >= ' ' || text[i] == '\t'; i++)
  ;
  process.stderr.write(text.slice(linestart,i))
  fputc('\n', stderr);

  // print caret
  for (var i = linestart; i!=tokstart; i++)
    process.stderr.write(text[i] == '\t' ? '\t' : ' ');
  console.error( "^");

  // print message and exit
  console.error("%s:%d: %s", file, line, msg);
  process.exit(1);
}

function isalnum(c) {
	return isalpha(c) || isdigit(c)
}

function isalpha(c) {
	return islower(c) || isupper(c)
}

function isdigit(c) {
	return '0' <= c && c <= '9'
}

function islower(c) {
	return 'a' <= c && c <= 'z'
}

function isspace(c) {
	switch (c) {
	case '\t':
	case '\n':
	case '\v':
	case '\f':
	case '\r':
	case ' ':
		return true
	}
}

function isupper(c) {
	return 'A' <= c && c <= 'Z'
}

exports.isalnum = isalnum
exports.isalpha = isalpha
exports.isdigit = isdigit
exports.islower = islower
exports.isspace = isspace
exports.isupper = isupper
