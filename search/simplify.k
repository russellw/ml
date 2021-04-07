(fn simplify(env a)
	(var op(hd a))
	(var args(map(\ x(simplify env x))(tl a)))
	(var a(cons op args))
	(var a
		(if (every num? args)
			(eval env	a)
			a))
	(match a
		a
	)
)

(assert= (simplify (here)#(+ 1 2))3)
