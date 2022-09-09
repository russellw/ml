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

namespace {
//symbols
struct Sym {
	SmallVector<GlobalValue*> defs;
	SmallVector<GlobalValue*> refs;

	void add(GlobalValue* a) {
		if (a->isDeclaration()) refs.push_back(a);
		else
			defs.push_back(a);
	}
};

DenseMap<StringRef, Sym*> syms;

//output
void h1(StringRef s) {
	outs() << "<h1 id=\"" << s << "\">" << s << "</h1>\n";
}

void h2(StringRef s) {
	outs() << "<h2 id=\"" << s << "\">" << s << "</h2>\n";
}

void h3(StringRef s) {
	outs() << "<h3 id=\"" << s << "\">" << s << "</h3>\n";
}

void href(StringRef tag, StringRef s) {
	outs() << tag << "<a href=\"#" << s << "\">" << s << "</a>\n";
}

void num(size_t n) {
	outs() << "<td style=\"text-align: right\">";
	if (n) outs() << n;
	outs() << '\n';
}

void globalValueTable() {
	outs() << "<table>\n";
	outs() << "<tr>\n";
	outs() << "<td>\n";
	outs() << "<td>Linkage\n";
	outs() << "<td>Type\n";
}

void print(GlobalValue& a) {
	outs() << "<tr>\n";
	if (a.hasExternalLinkage()) href("<td>", a.getName());
	else
		outs() << "<td>" << a.getName() << '\n';
	outs() << "<td>" << getLinkageName(a.getLinkage()) << '\n';
	outs() << "<td>" << *a.getValueType() << '\n';
}

void printSymVal(GlobalValue* a) {
	outs() << "<tr>\n";
	href("<td>", a->getParent()->getName());
	outs() << "<td>" << getLinkageName(a->getLinkage()) << '\n';
	outs() << "<td>" << *a->getValueType() << '\n';
}

void printSym(Sym* y) {
	outs() << "<table>\n";

	outs() << "<tr>\n";
	outs() << "<td colspan=3>Definitions\n";
	for (auto a: y->defs) printSymVal(a);

	outs() << "<tr>\n";
	outs() << "<td colspan=3>References\n";
	for (auto a: y->refs) printSymVal(a);

	outs() << "</table>\n";
}
} // namespace

int main(int argc, char** argv) {
	InitLLVM _(argc, argv);

	// Command line
	cl::list<std::string> files(cl::Positional, cl::OneOrMore, cl::desc("<files>"));
	cl::ParseCommandLineOptions(argc, argv, "Module cross-reference\n");

	//read files
	LLVMContext context;
	SmallVector<Module*, 0> modules;
	for (auto& file: files) {
		SMDiagnostic err;
		Module* m = parseIRFile(file, err, context).release();
		if (!m) {
			err.print(argv[0], errs());
			return 1;
		}
		modules.push_back(m);
		for (auto& kv: m->getValueSymbolTable()) {
			if (auto* a = dyn_cast<GlobalValue>(kv.getValue())) {
				if (!a->hasExternalLinkage()) continue;
				auto& y = syms[kv.getKey()];
				if (!y) y = new Sym;
				y->add(a);
				continue;
			}
			errs() << *kv.getValue();
			return 1;
		}
	}

	//output
	outs() << "<!DOCTYPE html>\n";
	outs() << "<html lang=\"en\">\n";
	outs() << "<meta charset=\"utf-8\"/>\n";
	outs() << "<style>\n";
	outs() << "td {\n";
	outs() << "padding-right: 10px;\n";
	outs() << "white-space: nowrap;\n";
	outs() << "}\n";
	outs() << "</style>\n";
	outs() << "<title>Module cross-reference</title>\n";

	//contents
	outs() << "<table>\n";

	outs() << "<tr>\n";
	href("<td>", "Modules");
	outs() << "<td style=\"text-align: right\">" << modules.size() << '\n';

	outs() << "<tr>\n";
	href("<td>", "Symbols");
	outs() << "<td style=\"text-align: right\">" << syms.size() << '\n';

	int ndups = 0;
	for (auto kv: syms)
		if (kv.second->defs.size() > 1) ++ndups;
	if (ndups) {
		outs() << "<tr>\n";
		href("<td>", "Duplicates");
		outs() << "<td style=\"text-align: right\">" << ndups << '\n';
	}

	outs() << "</table>\n";

	//modules
	outs() << "<hr>\n";
	h1("Modules");
	outs() << "<table>\n";
	outs() << "<tr>\n";
	outs() << "<td>\n";
	outs() << "<td style=\"text-align: right\">Vars\n";
	outs() << "<td style=\"text-align: right\">Fns\n";
	for (auto m: modules) {
		outs() << "<tr>\n";
		href("<td>", m->getName());
		num(m->getGlobalList().size());
		num(m->getFunctionList().size());
	}
	outs() << "</table>\n";

	for (auto m: modules) {
		outs() << "<hr>\n";
		h2(m->getName());

		h3("Vars");
		globalValueTable();
		for (auto& a: m->getGlobalList()) print(a);
		outs() << "</table>\n";

		h3("Fns");
		globalValueTable();
		for (auto& a: m->getFunctionList()) print(a);
		outs() << "</table>\n";
	}

	//symbols
	outs() << "<hr>\n";
	h1("Symbols");
	for (auto kv: syms) {
		h2(kv.first);
		printSym(kv.second);
	}

	//duplicates
	if (ndups) {
		outs() << "<hr>\n";
		h1("Duplicates");
		for (auto kv: syms) {
			if (kv.second->defs.size() < 2) continue;
			outs() << "<h2>" << kv.first << "</h2>\n";
			printSym(kv.second);
		}
	}

	return 0;
}
