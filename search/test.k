;logic
(assert 1)
(assert (not 0))

(assert=(and 0 0)0)
(assert=(and 0 1)0)
(assert=(and 1 0)0)
(assert=(and 1 1)1)

(assert=(or 0 0)0)
(assert=(or 0 1)1)
(assert=(or 1 0)1)
(assert=(or 1 1)1)

(assert=(if 1 (quote a) (quote b))(quote a))
(assert=(if 0 (quote a) (quote b))(quote b))

;comparison
(assert(= 5 5))
(assert(= 5.0 5.0))

(assert(/= 5 6))
(assert(/= 5.0 6.0))

(assert(< 1 2))
(assert(not(< 1 1)))
(assert(not(< 2 1)))
(assert(< 1/997 2/997))
(assert(not(< 1/997 1/997)))
(assert(not(< 2/997 1/997)))
(assert(< 1/2 2))
(assert(< -2 -1/2))
(assert(< 1.0 2.0))
(assert(not(< 1.0 1.0)))
(assert(not(< 2.0 1.0)))
(assert(< 1/2 2.0))
(assert(< -2.0 -1/2))

(assert(<= 1 2))
(assert(<= 1 1))
(assert(not(<= 2 1)))
(assert(<= 1/997 2/997))
(assert(<= 1/997 1/997))
(assert(not(<= 2/997 1/997)))
(assert(<= 1/2 2))
(assert(<= -2 -1/2))
(assert(<= 1.0 2.0))
(assert(<= 1.0 1.0))
(assert(not(<= 2.0 1.0)))
(assert(<= 1/2 2.0))
(assert(<= -2.0 -1/2))

;arithmetic
(assert=(+ 1 2)3)
(assert=(+ 1/7 2/7)3/7)
(assert=(+ 1/7 2)15/7)
(assert=(+ 1 2.5)3.5)

(assert=(- 1 2)-1)
(assert=(- 1/7 2/7)-1/7)
(assert=(- 1/7 2)-13/7)
(assert=(- 1 2.5)-1.5)

(assert=(* 12  12)144)
(assert=(* 12.0  12)144.0)
(assert=(* 12  12.0)144.0)
(assert=(* 12.0  12.0)144.0)
(assert=(*  1/2 1/2)1/4)
(assert=(*  1/2 0.5)0.25)
(assert=(* 0.5 1/2 )0.25)
(assert=(*  0.5 0.5)0.25)

(assert=(/ 1 3)1/3)
(assert=(/ 1 2.0)0.5)
