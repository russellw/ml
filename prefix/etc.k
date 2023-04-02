(fn map (f s)
	(if s
		(cons (f(hd s))(map f(tl s)))
		s))
(fn rnd-search(mk ev)
	(loop
		(= a (mk))
		(when (ev a)
			(= result a)
			(break))))
