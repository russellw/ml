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
