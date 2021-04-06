#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#endif
#include <errno.h>
#include <math.h>
#include <setjmp.h>
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
#include "cons.h"
#include "float.h"
#include "frame.h"
#include "int.h"
#include "rat.h"
#include "sym.h"

#include "keywords.h"

#include "term.h"

// parsers
#include "parser.h"

// algorithms
#include "eval.h"

// unit tests
#include "test.h"
