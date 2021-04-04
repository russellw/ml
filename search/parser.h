// input file
extern char *file;

// input text
extern char *txtstart;
extern char *txt;

// input token
extern char *tokstart;
extern int bufi;
extern int tok;
extern si tokterm;

// functions reusable by any parser
void readfile(void);
noret ferr(const char *msg);

// functions specific to this parser
void init_parser(void);
void parse(vec *v);
