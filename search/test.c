#include "stdafx.h"
// stdafx.h must be first
#include "main.h"

#ifdef DEBUG
void test(void) { assert(internz("abc") == internz("abc")); }
#endif
