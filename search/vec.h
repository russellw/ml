#define small 4

typedef struct  {
  uint32_t cap ;
  uint32_t n ;
  si *p ;
  si w[small];
}vec;

static void vinit(vec*v){
	v->cap=small;
	v->n=0;
	v->p=v->w;
}

void vpush(vec*v,si a);
void vfree(vec*v);
