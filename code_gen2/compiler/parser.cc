#include <olivine.h>

enum
{
	// SORT
	k_eq,
	k_ge,
	k_id,
	k_le,
	k_ne,
	///
};

namespace {
const char* txt;
int tok;

//tokenizer
void lex() {
	for (;;) {
		switch (*txt) {
		case ' ':
		case '\n':
		case '\r':
		case '\t':
			++txt;
			continue;
		case '!':
			switch (txt[1]) {
			case '=':
				txt += 2;
				tok = k_ne;
				return;
			}
			break;
		case '/':
			switch (txt[1]) {
			case '/':
				do ++txt;
				while (*txt != '\n' && *txt);
				continue;
			}
			break;
		case '<':
			switch (txt[1]) {
			case '=':
				txt += 2;
				tok = k_le;
				return;
			}
			break;
		case '=':
			switch (txt[1]) {
			case '=':
				txt += 2;
				tok = k_eq;
				return;
			}
			break;
		case '>':
			switch (txt[1]) {
			case '=':
				txt += 2;
				tok = k_ge;
				return;
			}
			break;
		case '_':
		case 'A':
		case 'a':
		case 'B':
		case 'b':
		case 'C':
		case 'c':
		case 'D':
		case 'd':
		case 'E':
		case 'e':
		case 'F':
		case 'f':
		case 'G':
		case 'g':
		case 'H':
		case 'h':
		case 'I':
		case 'i':
		case 'J':
		case 'j':
		case 'K':
		case 'k':
		case 'L':
		case 'l':
		case 'M':
		case 'm':
		case 'N':
		case 'n':
		case 'O':
		case 'o':
		case 'P':
		case 'p':
		case 'Q':
		case 'q':
		case 'R':
		case 'r':
		case 'S':
		case 's':
		case 'T':
		case 't':
		case 'U':
		case 'u':
		case 'V':
		case 'v':
		case 'W':
		case 'w':
		case 'X':
		case 'x':
		case 'Y':
		case 'y':
		case 'Z':
		case 'z':
		{
			auto s = txt;
			do ++txt;
			while (isId(*txt));
			tok = k_id;
			return;
		}
		}
		tok = *txt++;
		return;
	}
}
} // namespace

void parse(const char* file) {
	vector<char> text;
	readFile(file, text);
	txt = text.data();
	lex();
}
