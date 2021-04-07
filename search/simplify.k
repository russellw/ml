(fn simplify(env a)
	(var op(hd a))
	(var args(tl a))
	(var a
		(if (every num? args)
			(cons op(map(\ x(eval env x))args))
			a))
	(match a
		a
	)
)

(assert= (simplify (here)#(+ 1 2))3)
