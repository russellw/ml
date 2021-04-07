(fn simplify(env a)
	(var a
		(if
			(atom? a)
				a
			(cons(hd a)(map(\ x(simplify env x))(tl a)))))
	(eval  env a)
	{
	(var a
		(if (every num? args)
			(eval env	a)
			a))
	;(match a
		a
	;)
	}
)

(assert= (simplify (here)#(+ 1 2))3)
