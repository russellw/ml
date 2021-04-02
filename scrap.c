void pushc(int c) {
  if (bufi == sizeof buf - 1)
    err("token too long");
  buf[bufi++] = c;
}
