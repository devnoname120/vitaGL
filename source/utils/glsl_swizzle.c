#include "glsl_swizzle.h"

#include <stddef.h>
#include <string.h>

static int is_identifier_char(char value) {
	return (value >= 'a' && value <= 'z') ||
	       (value >= 'A' && value <= 'Z') ||
	       (value >= '0' && value <= '9') || value == '_';
}

static int is_space(char value) {
	return value == ' ' || value == '\t' || value == '\r' || value == '\n' ||
	       value == '\f' || value == '\v';
}

void glsl_translate_st_swizzles(char *source) {
	char *dot = strchr(source, '.');
	while (dot) {
		char *token = dot + 1;
		while (is_space(*token)) {
			token++;
		}

		char *end = token;
		while (is_identifier_char(*end)) {
			end++;
		}

		size_t length = end - token;
		if (length == 1) {
			if (token[0] == 's') {
				token[0] = 'x';
			} else if (token[0] == 't') {
				token[0] = 'y';
			}
		} else if (length == 2) {
			if (token[0] == 's' && token[1] == 't') {
				token[0] = 'x';
				token[1] = 'y';
			} else if (token[0] == 't' && token[1] == 's') {
				token[0] = 'y';
				token[1] = 'x';
			} else if (token[0] == 's' && token[1] == 's') {
				token[0] = token[1] = 'x';
			} else if (token[0] == 't' && token[1] == 't') {
				token[0] = token[1] = 'y';
			}
		}

		dot = strchr(end, '.');
	}
}
