enum {
	//SORT
  t_int,
  t_rat,
  t_sym,
  t_float,
  ///
};

#define term(p,t)((si)(p)+(t))
#define tag(a)((a)&7)

static Int* intptr(si a){
	assert(tag(a)==t_int);
	return(Int*)(a-t_int);
}
