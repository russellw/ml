'use strict'

// This should work with e.g.
// node --stack-size=10000
// but actually doesn't
// https://stackoverflow.com/questions/67305689/node-js-stack-size-silently-crashing
function f(...a) {
	console.log(a[0] + a[a.length - 1])
}

function g(a) {
	a = [...a]
	console.log(a[0] + a[a.length - 1])
}

var a = []
for (var i = 0; i < 1000000; i++) {
	a.push(i)
}
// f(...a)
g(a)
