
function transform(a,f){
	if(!a.length)return f(a)
	var r=[]
	Object.assign(r,a)
	for(var i=0;i<r.length;i++)
	r[i]=transform(r[i],f)
	return r
}

//transform
function t1(a){
	if(a.op=='integer')return integer(a.val+BigInt(1))
	return a
}

assert(eq(transform(a,t1), a))
assert(eq(transform(term('&&', bool(true), bool(true)),t1), term('&&', bool(true), bool(true))))
assert(eq(transform(call(f, [integer(1), integer(2)]),t1), call(f, [integer(2), integer(3)])))

function prnsolution(m) {
	var more
	for (var [k, v] of m) {
		if (!k.name) continue
		if (more) process.stdout.write(' ')
		more = true
		if (!v) process.stdout.write('-')
		process.stdout.write(k.name)
	}
	console.log()
}
