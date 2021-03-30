enum class type : si {
  none,

  Bool,
  Int,
  Rat,
  Real,
  Individual,

  max
};

// atomic types
const si atomicTypeBits = 12;
const si atomicTypes = 1 << atomicTypeBits;
extern const char *typeNames[atomicTypes];

struct sym;
type internType(sym *name);

// compound types
inline bool isCompound(type t) { return (si)t >= atomicTypes; }

struct tcompound {
  si n;
  type v[0];
};

inline tcompound *tcompoundp(type t) {
  assert(isCompound(t));
  return (tcompound *)t;
}

type internType(const vec<type> &v);
type internType(type r, type param1);
type internType(type rt, type param1, type param2);

// etc
enum class term : uint64_t;

void defaultType(type t, term a);
void requireType(type t, term a);

type typeof(term a);
type typeofNum(term a);

#ifdef DEBUG
void ck(type t);
#else
inline void ck(type t) {}
#endif

void print(type t);
