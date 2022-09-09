#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>

char code[1 << 20];

typedef unsigned char cell;
cell data[1 << 20];

int main(int argc, char** argv) {
	if (argc != 2) {
		fprintf(stderr, "Usage: brainf program.bf\n");
		return 1;
	}

	FILE* f = fopen(argv[1], "r");
	if (!f) {
		perror(argv[1]);
		return 1;
	}
	char* ip = code + 1;
	for (;;) {
		int c = fgetc(f);
		if (c < 0) break;
		if (ip == code + sizeof code - 1) {
			fprintf(stderr, "Code array overflow\n");
			return 1;
		}
		*ip++ = c;
	}
	fclose(f);

	ip = code + 1;
	cell* dp = data;
	size_t n;
loop:
	for (;;) switch (*ip++) {
		case '>':
			if (dp == data + sizeof data - 1) {
				fprintf(stderr, "Data array overflow\n");
				return 1;
			}
			dp++;
			break;
		case '<':
			if (dp == data) {
				fprintf(stderr, "Data array underflow\n");
				return 1;
			}
			dp--;
			break;
		case '+':
			++*dp;
			break;
		case '-':
			--*dp;
			break;
		case '.':
			putchar(*dp);
			break;
		case ',':
			*dp = getchar();
			break;
		case '[':
			if (*dp) break;
			n = 1;
			for (;;) switch (*ip++) {
				case '[':
					n++;
					break;
				case ']':
					n--;
					if (!n) goto loop;
					break;
				case 0:
					fprintf(stderr, "Unmatched '['\n");
					return 1;
				}
		case ']':
			if (!*dp) break;
			ip -= 2;
			n = 1;
			for (;;) switch (*ip--) {
				case ']':
					n++;
					break;
				case '[':
					n--;
					if (!n) {
						ip += 2;
						goto loop;
					}
					break;
				case 0:
					fprintf(stderr, "Unmatched ']'\n");
					return 1;
				}
		case 0:
			return 0;
		}
}
