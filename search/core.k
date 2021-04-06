(fn >( a b)(< b a))

(assert(> -1 -2))
(assert(> -1.9 -2))
(assert(> -1/9 -2))

(fn >=( a b)(<= b a))

(assert(>= -1 -2))
(assert(>= -1.9 -2))
(assert(>= -1/9 -2))
(assert(>= -1/9 -2/18))

(fn factorial( n)
	(if(<= n 1)
		1
		(* n(factorial(- n 1)))))

(assert=(factorial 0)1)
(assert=(factorial 1)1)
(assert=(factorial 2)2)
(assert=(factorial 3)6)
(assert=(factorial 4)24)
(assert=(factorial 5)120)

(fn atom? (a)(not(list? a)))

(assert(atom? 1))
(assert(atom? #foo))
(assert-not(atom? ()))

(fn num? (a)(or(exact? a)(inexact? a)))

(assert(num?	1))
(assert(num?	1/2))
(assert(num?	1.2))
(assert-not(num?	#a12))

(fn list1?(a)
	(and(list? a)a))

(assert-not(list1? 0))
(assert-not(list1? 9))
(assert-not(list1? ()))
(assert(list1? #(0)))
(assert(list1? #(9)))

(fn occurs (env a b)
	(if
		(= a b)1
		(can-get env a)(occurs env(get env a)b)
		(list1? b)(or(occurs env a(hd b))(occurs env a(tl b)))
	)
)

(assert(occurs() 3 3))
(assert(occurs() #a #a))
(assert-not(occurs() #a #b))
(assert(occurs() #a #( a b)))
(assert-not(occurs() #c #( a b)))
(assert-not(occurs#(((val c z))) #c #( a b)))
(assert(occurs#(((val c b))) #c #( a b)))
