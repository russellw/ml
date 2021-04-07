(fn simplify(env a)
		(if
			(atom? a)
				a
				(simplify-compound env a)))

(fn simplify-compound(env a)
	(var op(hd a))
	(var args(map(\ x(simplify env x))(tl a)))
	(var a(cons op args))
	(var a
		(if (every num? args)
			(eval env	a)
			a))
	(match a
		(+ ,x 0)	x
		(+ 0,x )	x
		(* ,x 1)	x
		(* 1,x )	x
		(* ,x 0)	0
		(* 0,x )	0
		(- ,x,x )	0
		a
	)
)

(assert= (simplify (here)#(+ 1 2))3)
(assert= (simplify (here)#(+ x y))#(+ x y))
(assert= (simplify (here)#(+ x 0))#x)
(assert= (simplify (here)#(+ 0 x ))#x)
(assert= (simplify (here)#(- x x ))0)
(assert= (simplify (here)#(- x x ))0)
(assert= (simplify (here)#(- x y ))#(- x y ))
