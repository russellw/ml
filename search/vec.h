#define vsmall 4

typedef struct {
  int cap;
  int n;
  si *p;
  si w[vsmall];
} vec;

static void vinit(vec *v) {
  v->cap = vsmall;
  v->n = 0;
  v->p = v->w;
}

void vpush(vec *v, si a);
void vfree(vec *v);
