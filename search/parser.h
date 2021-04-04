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

void readfile(void);
void init_parser(void);
void parse(vec *v);
