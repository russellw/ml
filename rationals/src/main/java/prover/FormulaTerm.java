package prover;

public class FormulaTerm extends Formula {
    private final Term term;

    public FormulaTerm(Term term) {
        this.term = term.unquantify();
    }

    public FormulaTerm(Term term, String name) {
        super(name);
        this.term = term.unquantify();
    }

    @Override
    public final Term term() {
        return term.quantify();
    }
}
