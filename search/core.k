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
(assert-not(occurs#((let c z)) #c #( a b)))
(assert(occurs#((let c b)) #c #( a b)))

(fn var? (a)
	(match a
		(var @s) 1))

(assert(var? #(var x)))
(assert(var? #(var x int)))
(assert-not(var? #(x var x )))

(fn unify-var(env a b)
	(if
		(can-get env a)
			(unify env (get env a) b)
		(can-get env b)
			(unify env a (get env b))
		(occurs env a b)
			0
		(cons #(let ,a ,b) env)))

(fn unify(env a b)
	(if
		(= env 0)
			env
		(= a b)
			env
		(var? a)
			(unify-var  env a  b)
		(var? b)
			(unify-var  env b a)
		(and(list1? a)(list1? b))
			(unify(unify env(hd a)(hd b))(tl a)(tl b))
		0))

; https://en.wikipedia.org/wiki/Unification_(computer_science)#Examples_of_syntactic_unification_of_first-order_terms
  ; Succeeds. (tautology)
(assert
	(list?
		(unify()#a #a)))

  ; a and b do not match
    (assert-not(list? (unify() #a #b )))

  ; Succeeds. (tautology)
    (assert (list?(unify() #(var x) #(var x) )))

  ; x is unified with the constant a
    (assert(list? (unify() #a #(var x) )))

  ; x and y are aliased
    (assert(list? (unify() #(var x) #(var y) )))

  ; function and constant symbols match, x is unified with the constant b
    (assert(list?
     (unify()
            #(f a (var x))
            #(f a b)
            )))

  ; f and g do not match
    (assert-not(list?
      (unify() #(f a) #(g a) )))

  ; x and y are aliased
    (assert(list?
     (unify()
            #(f (var x))
            #(f (var y))
            )))

  ; f and g do not match
    (assert-not(list?
      (unify()
             #(f (var x))
             #(g (var y))
             )))

  ; Fails. The f function symbols have different arity
    (assert-not(list?
      (unify()
             #(f (var x))
             #(f (var y) (var z))
             )))

  ; Unifies y with the term g(x)
    (assert(list?
     (unify()
            #(f (g (var x)))
            #(f (var y))
            )))

  ; Unifies x with constant a, and y with the term g(a)
    (assert(list?
     (unify()
            #(f (g (var x)) (var x))
            #(f (var y) a)
            )))

  ; Returns false in first-order logic and many modern Prolog dialects (enforced by the occurs check).
    (assert-not(list?
     (unify() #(var x) #(f (var x)) )))

  ; Both x and y are unified with the constant a
    (assert (list?(unify(unify() #(var x) #(var y) ) #(var y) #a )))

  ; As above (order of equations in set doesn't matter)
    (assert (list?(unify(unify() #a #(var y)) #(var x) #(var y) )))

  ; Fails. a and b do not match, so x can't be unified with both
    (assert-not (list?(unify(unify()# (var x) #a ) #b #(var x) )))

(fn any (f s)
	(if s
		(or(f(hd s))(any f(tl s)))))

(assert (any num? #(a b 3)))
(assert (any sym? #(a b 3)))
(assert-not (any sym? #()))

(fn every (f s)
	(if s
		(and(f(hd s))(every f(tl s)))
		1))

(assert-not (every num? #(a b 3)))
(assert-not (every sym? #(a b 3)))
(assert (every sym? #(a b c)))
(assert (every sym? #()))
