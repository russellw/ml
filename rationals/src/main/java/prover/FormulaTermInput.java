package prover;

public class FormulaTermInput extends FormulaTerm {
    private final String file;

    public FormulaTermInput(Term term, String name, String file) {
        super(term, name);
        this.file = file;
    }

    @Override
    public String file() {
        return file;
    }
}
