// input file
extern char *file;

// input text
extern char *txt;
extern char *txtstart;

// input token
extern char *tokstart;
extern int bufi;
extern int tok;
extern si tokterm;

void readfile(void);
void init_parser(void);
si parse(void);
