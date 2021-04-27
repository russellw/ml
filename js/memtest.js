// run with e.g.
// node --max-old-space-size=20000
'use strict'
var a = null
for (var i = 0; i < 1_000_000_000; i++) {
	if (i % 1_000_000 === 0) console.log(i)
	a = [a]
}
