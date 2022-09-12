#include <olivine.h>

#include <new>

#ifdef _WIN32
#include <windows.h>

namespace {
LONG WINAPI handler(struct _EXCEPTION_POINTERS* ExceptionInfo) {
	if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_STACK_OVERFLOW)
		WriteFile(GetStdHandle(STD_ERROR_HANDLE), "Stack overflow\n", 15, 0, 0);
	else {
		fprintf(stderr, "Exception code %lx\n", ExceptionInfo->ExceptionRecord->ExceptionCode);
				stackTrace();
	}
	ExitProcess(1);
}
} // namespace
#else
#include <unistd.h>
#endif

int main(int argc, const char** argv) {
	std::set_new_handler([]() {
		perror("new");
		exit(1);
	});
#ifdef _WIN32
	AddVectoredExceptionHandler(0, handler);
#endif
	return 0;
}
