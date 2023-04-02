;arithmetic
(assert(==(+ 1 2)3))
;functions
(fn square(a)
	(* a a))
(assert(==(square  3)9))
;higher-order functions
(assert(== (map square (quote (3 4))) (quote(9 16))))
;assignment
(= x 5)
(assert(== x 5))
