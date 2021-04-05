(def (> a b)(< b a))

(assert(> -1 -2))
(assert(> -1.9 -2))
(assert(> -1/9 -2))

(def (>= a b)(<= b a))

(assert(>= -1 -2))
(assert(>= -1.9 -2))
(assert(>= -1/9 -2))
(assert(>= -1/9 -2/18))
