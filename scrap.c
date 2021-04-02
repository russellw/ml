void pushc(int c) {
  if (bufi == sizeof buf - 1)
    err("token too long");
  buf[bufi++] = c;
}

void vprint(vec *v) {
	putchar('[');
	for(si i=0;i<v->n;i++){
		if(i)putchar(',');
			print(
	}
	puts("]");
}
