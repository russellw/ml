;arithmetic
(assert(==(+ 1 2)3))
;functions
(fn square(a)
	(* a a))
(assert(==(square  3)9))
;higher-order functions
(assert(== (map square (quote (3 4))) (quote(9 16))))
(assert(== (map (lambda(x)(+ x 1)) (quote (3 4))) (quote(4 5))))
;assignment
(= x 5)
(assert(== x 5))
;do
(do
	(= a 1)
	(= b 2))
(assert(== a 1))
(assert(== b 2))
;when
(when 1
	(= a 10)
	(= b 20))
(assert(== a 10))
(assert(== b 20))
;search
(assert
	(==
		(rnd-search
			(lambda()(rnd-int 5))
			(lambda(a)(== a 2))
		)
		2
	)
)
