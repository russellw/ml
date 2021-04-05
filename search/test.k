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

  ; floor division
  (assert=(div-f 5  3)  1)
  (assert=(div-f -5  3)  -2)
  (assert=(div-f 5  -3)  -2)
  (assert=(div-f -5  -3)  1)

  (assert=(div-f 5/997  3/997)  1)
  (assert=(div-f -5/997  3/997)  -2)
  (assert=(div-f 5/997  -3/997)  -2)
  (assert=(div-f -5/997  -3/997)  1)

  (assert=(div-f 10  -1/10)  -100)
  (assert=(div-f 1/10  10)  0)

  (assert=(rem-f 5  3)  2)
  (assert=(rem-f -5  3)  1)
  (assert=(rem-f 5  -3)  -1)
  (assert=(rem-f -5  -3)  -2)

  (assert=(rem-f 5/997  3/997) (* 2  997))
  (assert=(rem-f -5/997  3/997) (* 1  997))
  (assert=(rem-f 5/997  -3/997)  (* -1  997))
  (assert=(rem-f -5/997  -3/997)  (* -2  997))

  ; ceiling division
  (assert=(div-c 5  3)  2)
  (assert=(div-c -5  3)  -1)
  (assert=(div-c 5  -3)  -1)
  (assert=(div-c -5  -3)  2)

  (assert=(div-c 5/997  3/997)  2)
  (assert=(div-c -5/997  3/997)  -1)
  (assert=(div-c 5/997  -3/997)  -1)
  (assert=(div-c -5/997  -3/997)  2)

  (assert=(div-c 10  -1/10)  -100)
  (assert=(div-c 1/10  10)  1)

  (assert=(rem-c 5  3)  -1)
  (assert=(rem-c -5  3)  -2)
  (assert=(rem-c 5  -3)  2)
  (assert=(rem-c -5  -3)  1)

  (assert=(rem-c 5/997  3/997)  (* -1  997))
  (assert=(rem-c -5/997  3/997)  (* -2  997))
  (assert=(rem-c 5/997  -3/997) (* 2  997))
  (assert=(rem-c -5/997  -3/997) (* 1  997))

  ; truncating division
  (assert=(div-t 5  3)  1)
  (assert=(div-t -5  3)  -1)
  (assert=(div-t 5  -3)  -1)
  (assert=(div-t -5  -3)  1)

  (assert=(div-t 5/997  3/997)  1)
  (assert=(div-t -5/997  3/997)  -1)
  (assert=(div-t 5/997  -3/997)  -1)
  (assert=(div-t -5/997  -3/997)  1)

  (assert=(div-t 10  -1/10)  -100)
  (assert=(div-t 1/10  10)  0)

  (assert=(rem-t 5  3)  2)
  (assert=(rem-t -5  3)  -2)
  (assert=(rem-t 5  -3)  2)
  (assert=(rem-t -5  -3)  -2)

  (assert=(rem-t 5/997  3/997) (* 2  997))
  (assert=(rem-t -5/997  3/997)  (* -2   997))
  (assert=(rem-t 5/997  -3/997) (* 2  997))
  (assert=(rem-t -5/997  -3/997)  (* -2  997))

  ; euclidean division
  (assert=(div-e 7  3)  2)
  (assert=(div-e -7  3)  -3)
  (assert=(div-e 7  -3)  -2)
  (assert=(div-e -7  -3)  3)

  (assert=(div-e 7/997  3/997)  2)
  (assert=(div-e -7/997  3/997)  -3)
  (assert=(div-e 7/997  -3/997)  -2)
  (assert=(div-e -7/997  -3/997)  3)

  (assert=(div-e 10  -1/10)  -100)
  (assert=(div-e 1/10  10)  0)

  (assert=(rem-e 7  3)  1)
  (assert=(rem-e -7  3)  2)
  (assert=(rem-e 7  -3)  1)
  (assert=(rem-e -7  -3)  2)

  (assert=(rem-e 7/997  3/997) (* 1  997))
  (assert=(rem-e -7/997  3/997) (* 2  997))
  (assert=(rem-e 7/997  -3/997) (* 1  997))
  (assert=(rem-e -7/997  -3/997) (* 2  997))

  ; unary minus
  (assert=(minus 2.5)  -2.5)
  (assert=(minus 2)  -2)
  (assert=(minus -1/7)  1/7)

  ; ceiling
  (assert=(ceil 0)  0)
  (assert=(ceil 1/10)  1)
  (assert=(ceil 5/10)  1)
  (assert=(ceil 9/10)  1)
  (assert=(ceil -1/10)  0)
  (assert=(ceil -5/10)  0)
  (assert=(ceil -9/10)  0)

  (assert=(ceil 0)  0)
  (assert=(ceil 0.1)  1.)
  (assert=(ceil 0.5)  1.)
  (assert=(ceil 0.9)  1.)
  (assert=(ceil -0.1)  -0.0)
  (assert=(ceil -0.5)  -0.0)
  (assert=(ceil -0.9)  -0.0)

  ; floor
  (assert=(floor 0)  0)
  (assert=(floor 1/10)  0)
  (assert=(floor 5/10)  0)
  (assert=(floor 9/10)  0)
  (assert=(floor -1/10)  -1)
  (assert=(floor -5/10)  -1)
  (assert=(floor -9/10)  -1)

  (assert=(floor 0)  0)
  (assert=(floor 0.1)  0.)
  (assert=(floor 0.5)  0.)
  (assert=(floor 0.9)  0.)
  (assert=(floor -0.1)  -1.0)
  (assert=(floor -0.5)  -1.0)
  (assert=(floor -0.9)  -1.0)

  ; truncate
  (assert=(trunc 0)  0)
  (assert=(trunc 1/10)  0)
  (assert=(trunc 5/10)  0)
  (assert=(trunc 9/10)  0)
  (assert=(trunc -1/10)  0)
  (assert=(trunc -5/10)  0)
  (assert=(trunc -9/10)  0)

  (assert=(trunc 0)  0)
  (assert=(trunc 0.1)  0.)
  (assert=(trunc 0.5)  0.)
  (assert=(trunc 0.9)  0.)
  (assert=(trunc -0.1)  -0.0)
  (assert=(trunc -0.5)  -0.0)
  (assert=(trunc -0.9)  -0.0)

  ; round
  (assert=(round 0)  0)
  (assert=(round 1/10)  0)
  (assert=(round 5/10)  0)
  (assert=(round 9/10)  1)
  (assert=(round -1/10)  0)
  (assert=(round -5/10)  0)
  (assert=(round -9/10)  -1)
  (assert=(round 15/10)  2)
  (assert=(round 25/10)  2)
  (assert=(round 35/10)  4)
  (assert=(round 45/10)  4)

  (assert=(round 0)  0)
  (assert=(round 0.1)  0.)
  (assert=(round 0.9)  1.0)
  (assert=(round -0.1)  -0.0)
  (assert=(round -0.9)  -1.0)

  ; abs
  (assert=(abs 0)  0)
  (assert=(abs 1/10)  1/10)
  (assert=(abs 5/10)  5/10)
  (assert=(abs 9/10)  9/10)
  (assert=(abs -1/10)  1/10)
  (assert=(abs -5/10)  5/10)
  (assert=(abs -9/10)  9/10)

  (assert=(abs 0)  0)
  (assert=(abs 0.1)  0.1)
  (assert=(abs 0.5)  0.5)
  (assert=(abs 0.9)  0.9)
  (assert=(abs -0.1)  0.1)
  (assert=(abs -0.5)  0.5)
  (assert=(abs -0.9)  0.9)

;quote
(assert=`a a)

;nil is self-evaluating
(assert= () `())

;lists
(assert=`(a b c) (cons `a(cons `b(cons`c()))))
