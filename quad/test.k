{logic}
(assert 1)
(assert (not 0))

;comparison
(assert (== 10000 10000 10000))
(assert (not (== 10000 10000 10001)))
(assert (not (== 10000 10001 10000)))
(assert (not (== 10001 10000 10000)))
(assert (< -1 0 1))
(assert (not (< -1 0 0)))

;arithmetic
(assert (== (+ 1 2) 3))
(assert (== (+ 1 2 3) 6))

;functions
(fn square (a)
  (* a a))

(assert (== (square 3) 9))

;higher-order functions
(assert (== (map square (quote (3 4))) (quote (9 16))))
(assert
  (==
    (map
      (\ (x)
        (+ x 1))
      (quote (3 4)))
    (quote (4 5))))

;assignment
(= x 5)
(assert (== x 5))

;do
(do
  (= a 1)
  (= b 2))
(assert (== a 1))
(assert (== b 2))

;when
(when 1
  (= a 10)
  (= b 20))
(assert (== a 10))
(assert (== b 20))

;search
(assert
  (==
    (rnd-search
      (\ ()
        (randrange 5))
      (\ (a)
        (== a 2)))
    2))
(assert
  (<=
    0
    (rnd-search-best
      (\ ()
        (randrange 5))
      (\ (a)
        a))
    4))
