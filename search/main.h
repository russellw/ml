#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC_NEW
#endif
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <crtdbg.h>
#else
#define _CrtCheckMemory() 1
#endif

#include <gmp.h>
#include <xxhash.h>

#include "etc.h"
#include "vec.h"

// data
typedef struct {
  si n;

  // for the keyword system to work, the size of the declared character array
  // must be large enough to hold the longest keyword

  // for the system to work efficiently, the size of the whole structure must be
  // a power of 2

  // when symbols are allocated on the heap, the code doing the allocation is
  // responsible for allocating enough space to hold the corresponding strings
  char v[0x20 - sizeof(si)];
} sym;

#include "int.h"
#include "keywords.h"
#include "rat.h"
#include "float.h"
#include "sym.h"
#include "term.h"

// parsers
//#include "parser.h"

// algorithms
#include "eval.h"

// unit tests
#include "test.h"
