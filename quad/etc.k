(fn > (a b)
  (< b a))

(fn >= (a b)
  (<= b a))

(fn /= (a b)
  (not (== b a)))

(fn map (f s)
  (if s (cons (f (hd s)) (map f (tl s))) s))

(fn rnd-search (mk ev)
  (loop
    (= a (mk))
    (when (ev a)
      (= result a)
      (break))))

(fn rnd-search-best (mk ev)
  (= best -1)
  (loop
    (= a (mk))
    (= score (ev a))
    (when (> score best)
      (= result a)
      (= best score))))
