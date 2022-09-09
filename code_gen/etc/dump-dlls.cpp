#include <olivine.h>

// from AsmWriter.cpp
static std::string getLinkageName(GlobalValue::LinkageTypes LT) {
	switch (LT) {
	case GlobalValue::ExternalLinkage:
		return "external";
	case GlobalValue::PrivateLinkage:
		return "private";
	case GlobalValue::InternalLinkage:
		return "internal";
	case GlobalValue::LinkOnceAnyLinkage:
		return "linkonce";
	case GlobalValue::LinkOnceODRLinkage:
		return "linkonce_odr";
	case GlobalValue::WeakAnyLinkage:
		return "weak";
	case GlobalValue::WeakODRLinkage:
		return "weak_odr";
	case GlobalValue::CommonLinkage:
		return "common";
	case GlobalValue::AppendingLinkage:
		return "appending";
	case GlobalValue::ExternalWeakLinkage:
		return "extern_weak";
	case GlobalValue::AvailableExternallyLinkage:
		return "available_externally";
	}
	llvm_unreachable("invalid linkage");
}

static void PrintDLLStorageClass(GlobalValue::DLLStorageClassTypes SCT) {
	switch (SCT) {
	case GlobalValue::DefaultStorageClass:
		break;
	case GlobalValue::DLLImportStorageClass:
		outs() << "dllimport";
		break;
	case GlobalValue::DLLExportStorageClass:
		outs() << "dllexport";
		break;
	}
}

LLVMContext context;
StringMap<Module*> dllMap;
SmallVector<Module*, 0> dlls;

Module* getDLL(StringRef name) {
	auto& m = dllMap[name];
	if (!m) {
		m = new Module(name, context);
		dlls.push_back(m);
	}
	return m;
}

void readIR(StringRef dllName, MemoryBufferRef mb) {
	SMDiagnostic err;
	auto m = parseIR(mb, err, context);
	if (!m) {
		err.print(nullptr, errs());
		exit(1);
	}
	auto dll = getDLL(dllName);
	if (Linker::linkModules(*dll, std::move(m))) exit(1);
}

bool isMain(const Function& f) {
	return f.getName() == "main";
}

int main(int argc, char** argv) {
	InitLLVM _(argc, argv);

	// Command line
	cl::list<std::string> files(cl::Positional, cl::OneOrMore, cl::desc("<files>"));
	cl::ParseCommandLineOptions(argc, argv, "Dump symbols in IR-format DLLs\n");

	//read files
	for (const auto& file: files) {
		ExitOnError ExitOnErr(file + ": ");
		auto mb = ExitOnErr(errorOrToExpected(MemoryBuffer::getFileOrSTDIN(file)));

		//list file
		if (sys::path::extension(file) == ".tsv") {
			auto s = mb->getBufferStart();
			while (*s) {
				//DLL name
				auto t = s;
				while (*s != '\t') ++s;
				StringRef dllName(t, s - t);
				++s;

				//module name
				t = s;
				while (*(unsigned char*)s > '\r') ++s;
				std::string moduleName(t, s - t);
				if (*s == '\r') ++s;
				if (*s != '\n') {
					errs() << file << ": bad TSV file\n";
					return 1;
				}
				++s;

				//IR file
				ExitOnError ExitOnErr(moduleName + ": ");
				auto mb = ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(moduleName)));
				readIR(dllName, *mb);
			}
			continue;
		}

		//IR file
		readIR("", *mb);
	}

	//link DLLs
	for (auto m: dlls)
		for (auto& kv: m->getValueSymbolTable()) {
			if (auto a = dyn_cast<GlobalValue>(kv.getValue())) {
				if (a->isDeclaration()) continue;
				outs() << m->getName() << '\t';
				outs() << a->getName() << '\t';
				outs() << getLinkageName(a->getLinkage()) << '\t';
				PrintDLLStorageClass(a->getDLLStorageClass());
				outs() << '\n';
				continue;
			}
			errs() << *kv.getValue();
			return 1;
		}
	return 0;
}
