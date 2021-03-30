template <class T, si small = 4> struct vec {
  // types
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = si;
  using difference_type = intptr_t;
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  // data
  uint32_t cap = small;
  uint32_t n = 0;
  T *p = w;
  T w[small];

  // construct/copy/destroy
  vec() {}

  vec(si m) { resize(m); }

  vec(const vec &b) : n(b.n) {
    if (b.p == b.w) {
      memcpy(w, b.w, n * sizeof(T));
      return;
    }
    cap = b.cap;
    p = b.p;
    b.p = b.w;
  }

  ~vec() {
    if (p != w)
      free(p);
  }

  // iterators
  iterator begin() { return p; }
  const_iterator begin() const { return p; }

  iterator end() { return p + n; }
  const_iterator end() const { return p + n; }

  reverse_iterator rbegin() { return reverse_iterator(p + n); }
  const_reverse_iterator rbegin() const { return reverse_iterator(p + n); }

  reverse_iterator rend() { return reverse_iterator(p); }
  const_reverse_iterator rend() const { return const_reverse_iterator(p); }

  // capacity
  bool empty() const { return !n; }

  void resize(si m) {
    reserve(m);
    n = m;
  }

  void resize(si m, T a) {
    reserve(m);
    for (auto i = n; i < m; ++i)
      p[i] = a;
    n = m;
  }

  void reserve(si m) {
    if (m <= cap)
      return;
    cap = max(m, (si)cap * 2);
    if (p == w) {
      p = (T *)xmalloc(cap * sizeof *p);
      memcpy(p, w, n * sizeof(T));
      return;
    }
    p = (T *)xrealloc(p, cap * sizeof *p);
  }

  // element access
  T &operator[](si i) {
    assert(0 <= i);
    assert(i < n);
    return p[i];
  }

  const T &operator[](si i) const {
    assert(0 <= i);
    assert(i < n);
    return p[i];
  }

  T &front() {
    assert(n > 0);
    return *p;
  }

  const T &front() const {
    assert(n > 0);
    return *p;
  }

  T &back() {
    assert(n > 0);
    return p[n - 1];
  }

  const T &back() const {
    assert(n > 0);
    return p[n - 1];
  }

  // modifiers
  void push_back(T a) {
    reserve(n + 1);
    p[n++] = a;
  }

  void pop_back() {
    assert(n > 0);
    --n;
  }

  void insert(const_iterator position, T a) {
    assert(p <= position);
    assert(position <= end());
    auto i = position - p;

    reserve(n + 1);
    memmove(p + i + 1, p + i, (n - i) * sizeof(T));
    p[i] = a;
    ++n;
  }

  void insert(const_iterator position, T *first, T *last) {
    assert(p <= position);
    assert(position <= end());
    auto i = position - p;

    assert(first <= last);
    auto m = last - first;

    reserve(n + m);
    memmove(p + i + m, p + i, (n - i) * sizeof(T));
    memcpy(p + i, first, m * sizeof(T));
    n += m;
  }

  void erase(iterator position) { erase(position, position + 1); }

  void erase(iterator first, iterator last) {
    assert(p <= first);
    assert(first <= end());

    assert(p <= last);
    assert(last <= end());

    assert(first <= last);

    memmove(first, last, (end() - last) * sizeof(T));
    n -= last - first;
  }
};

template <class T> inline void print(const vec<T> &v) {
  putchar('[');
  auto n = v.n;
  for (si i = 0; i != n; ++i) {
    if (i)
      printf(", ");
    print(v[i]);
  }
  putchar(']');
}
