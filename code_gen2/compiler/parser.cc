#include <olivine.h>

enum
{
	// SORT
	k_and,
	k_eq,
	k_ge,
	k_id,
	k_le,
	k_ne,
	k_or,
	///
};

namespace {
const char* file;
vector<char> text;

//tokenizer
char* txt;

int tok;
char* tokStr;

[[noreturn]] void err(const char* msg) { ::err(file, text.data(), txt, msg); }

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
		case '#':
		eol:
			do ++txt;
			while (*txt != '\n' && *txt);
			continue;
		case '&':
			switch (txt[1]) {
			case '&':
				txt += 2;
				tok = k_and;
				return;
			}
			break;
		case '/':
			switch (txt[1]) {
			case '/':
				goto eol;
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
			tokStr = intern(s, txt - s);
			return;
		}
		case '|':
			switch (txt[1]) {
			case '|':
				txt += 2;
				tok = k_or;
				return;
			}
			break;
		case 0:
			tok = 0;
			return;
		}
		tok = *txt++;
		return;
	}
}

//parser
bool eat(int k) {
	if (tok == k) {
		lex();
		return 1;
	}
	return 0;
}

void expect(int k) {
	if (eat(k)) return;
	sprintf(buf, "expected '%c'", k);
	err(buf);
}

dyn id() {
	if (tok != k_id) err("expected identifier");
	dyn a(tokStr, t_sym);
	lex();
	return a;
}

//types
dyn typ() {
	if (tok != k_id) return list();
	dyn t(tokStr, t_sym);
	lex();
	return t;
}

dyn type() {
	dyn t = typ();
	if (t == list()) err("expected type");
	return t;
}

// Expressions.
dyn expr();

dyn primary() {
	switch (tok) {
	case '(':
	{
		lex();
		dyn a = expr();
		expect(')');
		return a;
	}
	case k_id:
		return id();
	}
	err("expected expression");
}

dyn postfix() {
	dyn a = primary();
	for (;;) switch (tok) {
		case '[':
			lex();
			a = list(s_subscript, a, expr());
			expect(']');
			break;
		case '(':
		{
			lex();
			vector<dyn> v(1, a);
			if (tok != ')') do
					v.push_back(expr());
				while (eat(','));
			expect(')');
			a = list(v);
			break;
		}
		default:
			return a;
		}
}

dyn prefix() {
	switch (tok) {
	case '!':
		lex();
		return list(s_not, prefix());
	}
	return postfix();
}

dyn expr() { return prefix(); }

//statements
dyn stmt() {
	if (tok == k_id) {
		auto k = keyword(tokStr);
		switch (k) {
		case s_return:
		{
			lex();
			if (eat(';')) return sym(k);
			dyn a = expr();
			expect(';');
			return list(k, a);
		}
		}
	}
	dyn a = expr();
	expect(';');
	return a;
}

//declarations
dyn decl() {
	dyn t = type();
	dyn name = id();
	expect('(');
	vector<dyn> params;
	if (tok != ')') do {
			dyn u = type();
			params.push_back(list(u, id()));
		} while (eat(','));
	expect(')');
	return list(s_fn, t, name, list(params), stmt());
}
} // namespace

void parse(const char* f) {
	file = f;
	readFile(f, text);
	txt = text.data();
	lex();
	vector<dyn> v;
	while (tok) v.push_back(decl());
}
